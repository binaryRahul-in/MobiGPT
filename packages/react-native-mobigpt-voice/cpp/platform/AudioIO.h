// MobiGPT Voice — platform audio abstraction.
//
// Implementations:
//   android/src/main/cpp/AndroidAudio.cpp  AudioTrack / AudioRecord via JNI
//   ios/AppleAudio.mm                      AVAudioEngine
//   cpp/platform/NullAudio.cpp             host builds / unit tests
//
// Audio never crosses into JavaScript: the engine writes PCM straight into
// the platform sink from its worker thread.
#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>

namespace mobigpt::audio {

class AudioOutput {
 public:
  virtual ~AudioOutput() = default;
  // Opens a mono float stream. Returns false (and sets error()) on failure.
  virtual bool start(int sampleRate) = 0;
  // Blocking write; returns false if the stream was stopped / failed.
  virtual bool write(const float* samples, size_t count) = 0;
  // Waits for queued audio to finish, then closes the stream.
  virtual void drain() = 0;
  // Stops immediately (drops queued audio).
  virtual void stop() = 0;
  virtual std::string error() const { return {}; }
};

class AudioInput {
 public:
  using Callback = std::function<void(const float* samples, size_t count)>;
  virtual ~AudioInput() = default;
  // Starts capture; `cb` runs on a platform audio thread. The implementation
  // may pick a different rate than requested — check sampleRate().
  virtual bool start(int preferredSampleRate, Callback cb) = 0;
  virtual void stop() = 0;
  virtual int sampleRate() const = 0;
  virtual std::string error() const { return {}; }
};

std::unique_ptr<AudioOutput> createAudioOutput();
std::unique_ptr<AudioInput> createAudioInput();

// Optional test hook (host only): route output into a callback.
void setHostOutputTap(std::function<void(const float*, size_t, int)> tap);

}  // namespace mobigpt::audio
