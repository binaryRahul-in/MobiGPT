// Host (desktop / CI) audio backend: output can be tapped by tests, input
// produces silence. Compiled only when neither Android nor Apple backends
// are available.
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>

#include "AudioIO.h"

namespace mobigpt::audio {

namespace {
std::mutex gTapMutex;
std::function<void(const float*, size_t, int)> gTap;

class NullOutput final : public AudioOutput {
 public:
  bool start(int sampleRate) override {
    sr_ = sampleRate;
    running_ = true;
    return true;
  }
  bool write(const float* s, size_t n) override {
    if (!running_) return false;
    std::lock_guard<std::mutex> lk(gTapMutex);
    if (gTap) gTap(s, n, sr_);
    return true;
  }
  void drain() override { running_ = false; }
  void stop() override { running_ = false; }

 private:
  int sr_ = 0;
  std::atomic<bool> running_{false};
};

class NullInput final : public AudioInput {
 public:
  ~NullInput() override { stop(); }
  bool start(int sr, Callback cb) override {
    sr_ = sr;
    running_ = true;
    thread_ = std::thread([this, cb] {
      std::vector<float> zeros(static_cast<size_t>(sr_ / 50), 0.0f);  // 20 ms blocks
      while (running_) {
        cb(zeros.data(), zeros.size());
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }
    });
    return true;
  }
  void stop() override {
    running_ = false;
    if (thread_.joinable()) thread_.join();
  }
  int sampleRate() const override { return sr_; }

 private:
  int sr_ = 16000;
  std::atomic<bool> running_{false};
  std::thread thread_;
};
}  // namespace

std::unique_ptr<AudioOutput> createAudioOutput() { return std::make_unique<NullOutput>(); }
std::unique_ptr<AudioInput> createAudioInput() { return std::make_unique<NullInput>(); }

void setHostOutputTap(std::function<void(const float*, size_t, int)> tap) {
  std::lock_guard<std::mutex> lk(gTapMutex);
  gTap = std::move(tap);
}

}  // namespace mobigpt::audio
