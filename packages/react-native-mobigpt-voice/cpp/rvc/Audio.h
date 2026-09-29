// MobiGPT Voice — audio primitives (WAV I/O, resampling, levels).
// Pure C++17, no platform dependencies so it can be unit-tested on the host.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mobigpt::rvc {

struct AudioBuffer {
  std::vector<float> samples;  // mono, [-1, 1]
  int sampleRate = 0;

  double durationSeconds() const {
    return sampleRate > 0 ? static_cast<double>(samples.size()) / sampleRate : 0.0;
  }
};

// Reads a RIFF/WAVE file (PCM 8/16/24/32-bit int or 32/64-bit float, any
// channel count). Multi-channel audio is down-mixed to mono.
// Throws std::runtime_error on malformed / unsupported input.
AudioBuffer readWav(const std::string& path);

// Parses WAV bytes already in memory (same rules as readWav).
AudioBuffer parseWav(const uint8_t* data, size_t size);

// Writes mono 16-bit PCM WAV (the most widely playable format).
void writeWavPcm16(const std::string& path, const float* samples, size_t count, int sampleRate);

// Incremental WAV writer: header is patched on close() so audio can be
// streamed to disk chunk by chunk without holding it all in memory.
class WavStreamWriter {
 public:
  WavStreamWriter() = default;
  ~WavStreamWriter();
  WavStreamWriter(const WavStreamWriter&) = delete;
  WavStreamWriter& operator=(const WavStreamWriter&) = delete;

  void open(const std::string& path, int sampleRate);
  void write(const float* samples, size_t count);
  void close();
  bool isOpen() const { return file_ != nullptr; }
  size_t framesWritten() const { return frames_; }

 private:
  void* file_ = nullptr;  // FILE*
  int sampleRate_ = 0;
  size_t frames_ = 0;
};

// Band-limited resampler (Kaiser-windowed sinc). Quality is plenty for
// speech front-ends (HuBERT/RMVPE run at 16 kHz) and it is streaming-safe
// when used through StreamResampler.
std::vector<float> resample(const float* in, size_t count, int srcRate, int dstRate);

class StreamResampler {
 public:
  StreamResampler(int srcRate, int dstRate, int halfTaps = 16);
  // Appends converted samples to `out`.
  void process(const float* in, size_t count, std::vector<float>& out);
  // Emits the remaining tail (zero-padded) — call once at end of stream.
  void flush(std::vector<float>& out);
  int srcRate() const { return src_; }
  int dstRate() const { return dst_; }

 private:
  float sampleAt(double pos) const;
  int src_, dst_, halfTaps_;
  int span_;          // kernel half-width in *input* samples (stretched when downsampling)
  double step_;       // input samples per output sample
  double pos_ = 0.0;  // next output position, in input-sample units relative to history_[0]
  double cutoff_;
  std::vector<float> history_;
  float kaiserBeta_ = 8.0f;
  std::vector<float> kaiser_;  // window lookup over |x| in [0, 1]
};

float rms(const float* x, size_t n);
float peak(const float* x, size_t n);
void clipInPlace(float* x, size_t n, float limit = 1.0f);

}  // namespace mobigpt::rvc
