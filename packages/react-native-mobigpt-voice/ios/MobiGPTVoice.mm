#import "MobiGPTVoice.h"

#import <AVFoundation/AVFoundation.h>

#include <memory>
#include <vector>

#include "Audio.h"
#include "jsi/VoiceJSI.h"

@interface MobiGPTVoice () <AVSpeechSynthesizerDelegate>
@end

@implementation MobiGPTVoice {
  AVSpeechSynthesizer *_synth;
  BOOL _jsiInstalled;
}

RCT_EXPORT_MODULE(MobiGPTVoice)

+ (BOOL)requiresMainQueueSetup {
  return NO;
}

// New architecture: React Native calls this with the live runtime as soon as
// the TurboModule is created, so `install()` only has to report success.
- (void)installJSIBindingsWithRuntime:(facebook::jsi::Runtime &)runtime
                          callInvoker:(const std::shared_ptr<facebook::react::CallInvoker> &)callInvoker {
  mobigpt::jsi_bindings::install(runtime, callInvoker);
  _jsiInstalled = YES;
}

- (void)invalidate {
  mobigpt::jsi_bindings::invalidate();
}

RCT_EXPORT_METHOD(install : (RCTPromiseResolveBlock)resolve reject : (RCTPromiseRejectBlock)reject) {
  resolve(@(_jsiInstalled));
}

RCT_EXPORT_METHOD(synthesizeSpeech : (NSString *)text outputPath : (NSString *)outputPath language : (NSString *)language
                  voiceId : (NSString *)voiceId rate : (double)rate pitch : (double)pitch resolve : (RCTPromiseResolveBlock)resolve
                  reject : (RCTPromiseRejectBlock)reject) {
  AVSpeechUtterance *u = [AVSpeechUtterance speechUtteranceWithString:text];
  AVSpeechSynthesisVoice *voice = nil;
  if (voiceId.length > 0) voice = [AVSpeechSynthesisVoice voiceWithIdentifier:voiceId];
  if (!voice && language.length > 0) voice = [AVSpeechSynthesisVoice voiceWithLanguage:language];
  if (voice) u.voice = voice;
  // AVSpeech rate is 0..1 with 0.5 = normal; map the Android-style multiplier.
  const float r = rate > 0 ? (float)rate : 1.0f;
  u.rate = MIN(AVSpeechUtteranceMaximumSpeechRate, MAX(AVSpeechUtteranceMinimumSpeechRate, AVSpeechUtteranceDefaultSpeechRate * r));
  u.pitchMultiplier = pitch > 0 ? MIN(2.0f, MAX(0.5f, (float)pitch)) : 1.0f;

  if (!_synth) _synth = [AVSpeechSynthesizer new];
  auto samples = std::make_shared<std::vector<float>>();
  __block double sampleRate = 0;
  __block BOOL finished = NO;
  std::string path = outputPath.UTF8String;
  [_synth writeUtterance:u
        toBufferCallback:^(AVAudioBuffer *_Nonnull buffer) {
          if (finished) return;
          AVAudioPCMBuffer *pcm = (AVAudioPCMBuffer *)buffer;
          if (![pcm isKindOfClass:[AVAudioPCMBuffer class]] || pcm.frameLength == 0) {
            finished = YES;
            try {
              mobigpt::rvc::writeWavPcm16(path, samples->data(), samples->size(), (int)(sampleRate > 0 ? sampleRate : 22050));
              resolve(@{@"path" : outputPath, @"sampleRate" : @(sampleRate), @"seconds" : @(sampleRate > 0 ? samples->size() / sampleRate : 0)});
            } catch (const std::exception &e) {
              reject(@"E_TTS", [NSString stringWithUTF8String:e.what()], nil);
            }
            return;
          }
          sampleRate = pcm.format.sampleRate;
          const AVAudioFrameCount n = pcm.frameLength;
          const size_t base = samples->size();
          samples->resize(base + n);
          if (pcm.floatChannelData) {
            memcpy(samples->data() + base, pcm.floatChannelData[0], n * sizeof(float));
          } else if (pcm.int16ChannelData) {
            for (AVAudioFrameCount i = 0; i < n; ++i) (*samples)[base + i] = pcm.int16ChannelData[0][i] / 32768.0f;
          } else if (pcm.int32ChannelData) {
            for (AVAudioFrameCount i = 0; i < n; ++i) (*samples)[base + i] = pcm.int32ChannelData[0][i] / 2147483648.0f;
          }
        }];
}

RCT_EXPORT_METHOD(listTtsVoices : (RCTPromiseResolveBlock)resolve reject : (RCTPromiseRejectBlock)reject) {
  NSMutableArray *out = [NSMutableArray new];
  for (AVSpeechSynthesisVoice *v in [AVSpeechSynthesisVoice speechVoices]) {
    [out addObject:@{
      @"id" : v.identifier,
      @"name" : v.name,
      @"language" : v.language,
      @"quality" : @(v.quality == AVSpeechSynthesisVoiceQualityEnhanced ? 400 : 300),
      @"requiresNetwork" : @NO,
    }];
  }
  resolve(out);
}

RCT_EXPORT_METHOD(decodeAudioToWav : (NSString *)inputPath outputPath : (NSString *)outputPath resolve : (RCTPromiseResolveBlock)
                      resolve reject : (RCTPromiseRejectBlock)reject) {
  dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
    NSURL *url = [inputPath hasPrefix:@"file://"] ? [NSURL URLWithString:inputPath] : [NSURL fileURLWithPath:inputPath];
    NSError *err = nil;
    AVAudioFile *file = [[AVAudioFile alloc] initForReading:url error:&err];
    if (!file) {
      reject(@"E_DECODE", err.localizedDescription ?: @"cannot open audio file", err);
      return;
    }
    AVAudioFormat *fmt = file.processingFormat;  // float32, deinterleaved
    const AVAudioFrameCount block = 16384;
    AVAudioPCMBuffer *buf = [[AVAudioPCMBuffer alloc] initWithPCMFormat:fmt frameCapacity:block];
    try {
      mobigpt::rvc::WavStreamWriter w;
      w.open(outputPath.UTF8String, (int)fmt.sampleRate);
      std::vector<float> mono(block);
      size_t total = 0;
      while (file.framePosition < file.length) {
        if (![file readIntoBuffer:buf frameCount:block error:&err] || buf.frameLength == 0) break;
        const AVAudioChannelCount ch = fmt.channelCount;
        for (AVAudioFrameCount i = 0; i < buf.frameLength; ++i) {
          float acc = 0;
          for (AVAudioChannelCount c = 0; c < ch; ++c) acc += buf.floatChannelData[c][i];
          mono[i] = acc / ch;
        }
        w.write(mono.data(), buf.frameLength);
        total += buf.frameLength;
        if (total > (size_t)fmt.sampleRate * 15 * 60) throw std::runtime_error("audio longer than 15 minutes");
      }
      w.close();
      resolve(@{@"path" : outputPath, @"sampleRate" : @(fmt.sampleRate), @"seconds" : @(total / fmt.sampleRate)});
    } catch (const std::exception &e) {
      reject(@"E_DECODE", [NSString stringWithUTF8String:e.what()], nil);
    }
  });
}

#ifdef RCT_NEW_ARCH_ENABLED
- (std::shared_ptr<facebook::react::TurboModule>)getTurboModule:(const facebook::react::ObjCTurboModule::InitParams &)params {
  return std::make_shared<facebook::react::NativeMobiGPTVoiceSpecJSI>(params);
}
#endif

@end
