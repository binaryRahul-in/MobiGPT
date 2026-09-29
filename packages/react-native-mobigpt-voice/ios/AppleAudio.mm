// iOS audio backend (AVAudioEngine). The RVC worker thread writes float PCM
// straight into an AVAudioPlayerNode; the mic tap feeds the engine's ring
// buffer. Nothing is routed through JavaScript.
#import <AVFoundation/AVFoundation.h>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <string>

#include "AudioIO.h"

namespace mobigpt::audio {

namespace {

bool configureSession(bool record, std::string& err) {
  AVAudioSession *s = [AVAudioSession sharedInstance];
  NSError *e = nil;
  if (record) {
    if (s.recordPermission != AVAudioSessionRecordPermissionGranted) {
      dispatch_semaphore_t sem = dispatch_semaphore_create(0);
      __block BOOL granted = NO;
      [s requestRecordPermission:^(BOOL ok) {
        granted = ok;
        dispatch_semaphore_signal(sem);
      }];
      dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, 60 * NSEC_PER_SEC));
      if (!granted) {
        err = "microphone permission denied";
        return false;
      }
    }
    [s setCategory:AVAudioSessionCategoryPlayAndRecord
              mode:AVAudioSessionModeDefault
           options:AVAudioSessionCategoryOptionDefaultToSpeaker | AVAudioSessionCategoryOptionAllowBluetoothA2DP
             error:&e];
  } else if (s.category != AVAudioSessionCategoryPlayAndRecord) {
    [s setCategory:AVAudioSessionCategoryPlayback error:&e];
  }
  [s setActive:YES error:&e];
  if (e) {
    err = e.localizedDescription.UTF8String;
    return false;
  }
  return true;
}

class EngineOutput final : public AudioOutput {
 public:
  ~EngineOutput() override { stop(); }

  bool start(int sampleRate) override {
    if (!configureSession(false, err_)) return false;
    engine_ = [AVAudioEngine new];
    player_ = [AVAudioPlayerNode new];
    format_ = [[AVAudioFormat alloc] initStandardFormatWithSampleRate:sampleRate channels:1];
    [engine_ attachNode:player_];
    [engine_ connect:player_ to:engine_.mainMixerNode format:format_];
    NSError *e = nil;
    if (![engine_ startAndReturnError:&e]) {
      err_ = e.localizedDescription.UTF8String ?: "AVAudioEngine failed to start";
      return false;
    }
    [player_ play];
    slots_ = dispatch_semaphore_create(kMaxQueued);
    running_ = true;
    return true;
  }

  bool write(const float *s, size_t n) override {
    while (n > 0 && running_) {
      const AVAudioFrameCount k = (AVAudioFrameCount)std::min<size_t>(n, kBlock);
      // Back-pressure: at most kMaxQueued buffers (~0.3 s) in flight.
      dispatch_semaphore_wait(slots_, DISPATCH_TIME_FOREVER);
      if (!running_) return false;
      AVAudioPCMBuffer *buf = [[AVAudioPCMBuffer alloc] initWithPCMFormat:format_ frameCapacity:k];
      buf.frameLength = k;
      memcpy(buf.floatChannelData[0], s, k * sizeof(float));
      pending_++;
      dispatch_semaphore_t slots = slots_;
      std::atomic<int> *pending = &pending_;
      [player_ scheduleBuffer:buf
            completionHandler:^{
              pending->fetch_sub(1);
              dispatch_semaphore_signal(slots);
            }];
      s += k;
      n -= k;
    }
    return running_;
  }

  void drain() override {
    for (int i = 0; i < 400 && pending_.load() > 0 && running_; ++i) [NSThread sleepForTimeInterval:0.01];
    stop();
  }

  void stop() override {
    if (!engine_) return;
    running_ = false;
    for (int i = 0; i < kMaxQueued; ++i) dispatch_semaphore_signal(slots_);
    [player_ stop];
    [engine_ stop];
    engine_ = nil;
    player_ = nil;
  }

  std::string error() const override { return err_; }

 private:
  static constexpr size_t kBlock = 4800;
  static constexpr int kMaxQueued = 3;
  AVAudioEngine *engine_ = nil;
  AVAudioPlayerNode *player_ = nil;
  AVAudioFormat *format_ = nil;
  dispatch_semaphore_t slots_ = nil;
  std::atomic<bool> running_{false};
  std::atomic<int> pending_{0};
  std::string err_;
};

class EngineInput final : public AudioInput {
 public:
  ~EngineInput() override { stop(); }

  bool start(int, Callback cb) override {
    if (!configureSession(true, err_)) return false;
    engine_ = [AVAudioEngine new];
    AVAudioInputNode *in = engine_.inputNode;
    AVAudioFormat *fmt = [in outputFormatForBus:0];
    if (fmt.sampleRate <= 0 || fmt.channelCount == 0) {
      err_ = "no microphone input available";
      return false;
    }
    rate_ = (int)fmt.sampleRate;
    auto callback = std::make_shared<Callback>(std::move(cb));
    [in installTapOnBus:0
             bufferSize:(AVAudioFrameCount)(fmt.sampleRate / 50)
                 format:fmt
                  block:^(AVAudioPCMBuffer *buffer, AVAudioTime *when) {
                    if (buffer.floatChannelData && buffer.frameLength > 0) (*callback)(buffer.floatChannelData[0], buffer.frameLength);
                  }];
    NSError *e = nil;
    if (![engine_ startAndReturnError:&e]) {
      err_ = e.localizedDescription.UTF8String ?: "AVAudioEngine input failed";
      [in removeTapOnBus:0];
      engine_ = nil;
      return false;
    }
    return true;
  }

  void stop() override {
    if (!engine_) return;
    [engine_.inputNode removeTapOnBus:0];
    [engine_ stop];
    engine_ = nil;
  }

  int sampleRate() const override { return rate_; }
  std::string error() const override { return err_; }

 private:
  AVAudioEngine *engine_ = nil;
  int rate_ = 48000;
  std::string err_;
};

}  // namespace

std::unique_ptr<AudioOutput> createAudioOutput() { return std::make_unique<EngineOutput>(); }
std::unique_ptr<AudioInput> createAudioInput() { return std::make_unique<EngineInput>(); }
void setHostOutputTap(std::function<void(const float *, size_t, int)>) {}

}  // namespace mobigpt::audio
