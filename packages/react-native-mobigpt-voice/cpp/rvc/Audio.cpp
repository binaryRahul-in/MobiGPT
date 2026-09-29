#include "Audio.h"
#include "Dsp.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace mobigpt::rvc {

namespace {

uint16_t rd16(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }
uint32_t rd32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

void wr16(FILE* f, uint16_t v) {
  uint8_t b[2] = {static_cast<uint8_t>(v & 0xff), static_cast<uint8_t>(v >> 8)};
  fwrite(b, 1, 2, f);
}
void wr32(FILE* f, uint32_t v) {
  uint8_t b[4] = {static_cast<uint8_t>(v & 0xff), static_cast<uint8_t>((v >> 8) & 0xff),
                  static_cast<uint8_t>((v >> 16) & 0xff), static_cast<uint8_t>(v >> 24)};
  fwrite(b, 1, 4, f);
}

void writeHeader(FILE* f, int sampleRate, uint32_t dataBytes) {
  fwrite("RIFF", 1, 4, f);
  wr32(f, 36 + dataBytes);
  fwrite("WAVE", 1, 4, f);
  fwrite("fmt ", 1, 4, f);
  wr32(f, 16);
  wr16(f, 1);  // PCM
  wr16(f, 1);  // mono
  wr32(f, static_cast<uint32_t>(sampleRate));
  wr32(f, static_cast<uint32_t>(sampleRate) * 2);
  wr16(f, 2);
  wr16(f, 16);
  fwrite("data", 1, 4, f);
  wr32(f, dataBytes);
}

inline int16_t toPcm16(float v) {
  v = std::max(-1.0f, std::min(1.0f, v));
  return static_cast<int16_t>(std::lrint(v * 32767.0f));
}

double besselI0(double x) {
  double sum = 1.0, term = 1.0, k = 1.0;
  const double q = x * x / 4.0;
  while (term > 1e-12 * sum) {
    term *= q / (k * k);
    sum += term;
    k += 1.0;
  }
  return sum;
}

}  // namespace

AudioBuffer parseWav(const uint8_t* data, size_t size) {
  if (size < 12 || std::memcmp(data, "RIFF", 4) != 0 || std::memcmp(data + 8, "WAVE", 4) != 0) {
    throw std::runtime_error("not a RIFF/WAVE file");
  }
  uint16_t format = 0, channels = 0, bits = 0;
  uint32_t rate = 0;
  const uint8_t* pcm = nullptr;
  size_t pcmBytes = 0;
  size_t off = 12;
  while (off + 8 <= size) {
    const uint8_t* ck = data + off;
    uint32_t ckSize = rd32(ck + 4);
    const uint8_t* body = ck + 8;
    size_t avail = size - off - 8;
    if (std::memcmp(ck, "fmt ", 4) == 0) {
      if (ckSize < 16 || avail < 16) throw std::runtime_error("bad fmt chunk");
      format = rd16(body);
      channels = rd16(body + 2);
      rate = rd32(body + 4);
      bits = rd16(body + 14);
      if (format == 0xFFFE && ckSize >= 26 && avail >= 26) format = rd16(body + 24);
    } else if (std::memcmp(ck, "data", 4) == 0) {
      pcm = body;
      // Streaming writers sometimes leave 0 / 0xFFFFFFFF as the size.
      pcmBytes = (ckSize == 0 || ckSize > avail) ? avail : ckSize;
      break;
    }
    off += 8 + ckSize + (ckSize & 1);
  }
  if (!pcm || channels == 0 || rate == 0) throw std::runtime_error("WAV missing fmt/data chunk");
  const bool isFloat = format == 3;
  if (!(format == 1 || isFloat)) throw std::runtime_error("unsupported WAV encoding (need PCM or float)");
  if (isFloat && bits != 32 && bits != 64) throw std::runtime_error("unsupported float WAV bit depth");
  if (!isFloat && bits != 8 && bits != 16 && bits != 24 && bits != 32) {
    throw std::runtime_error("unsupported PCM WAV bit depth");
  }

  const size_t bps = bits / 8;
  const size_t frameBytes = bps * channels;
  const size_t frames = pcmBytes / frameBytes;
  AudioBuffer out;
  out.sampleRate = static_cast<int>(rate);
  out.samples.resize(frames);
  for (size_t i = 0; i < frames; ++i) {
    double acc = 0.0;
    for (uint16_t c = 0; c < channels; ++c) {
      const uint8_t* s = pcm + i * frameBytes + c * bps;
      double v = 0.0;
      if (isFloat) {
        if (bits == 32) {
          float f;
          std::memcpy(&f, s, 4);
          v = f;
        } else {
          double d;
          std::memcpy(&d, s, 8);
          v = d;
        }
      } else if (bits == 8) {
        v = (static_cast<int>(s[0]) - 128) / 128.0;
      } else if (bits == 16) {
        v = static_cast<int16_t>(rd16(s)) / 32768.0;
      } else if (bits == 24) {
        int32_t x = (s[0] | (s[1] << 8) | (s[2] << 16));
        if (x & 0x800000) x |= ~0xFFFFFF;
        v = x / 8388608.0;
      } else {
        v = static_cast<int32_t>(rd32(s)) / 2147483648.0;
      }
      acc += v;
    }
    out.samples[i] = static_cast<float>(acc / channels);
  }
  return out;
}

AudioBuffer readWav(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  return parseWav(bytes.data(), bytes.size());
}

void writeWavPcm16(const std::string& path, const float* samples, size_t count, int sampleRate) {
  WavStreamWriter w;
  w.open(path, sampleRate);
  w.write(samples, count);
  w.close();
}

WavStreamWriter::~WavStreamWriter() {
  try {
    close();
  } catch (...) {
  }
}

void WavStreamWriter::open(const std::string& path, int sampleRate) {
  close();
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write " + path);
  file_ = f;
  sampleRate_ = sampleRate;
  frames_ = 0;
  writeHeader(f, sampleRate, 0);
}

void WavStreamWriter::write(const float* samples, size_t count) {
  if (!file_) throw std::runtime_error("WavStreamWriter not open");
  FILE* f = static_cast<FILE*>(file_);
  std::vector<uint8_t> buf(count * 2);
  for (size_t i = 0; i < count; ++i) {
    int16_t v = toPcm16(samples[i]);
    buf[2 * i] = static_cast<uint8_t>(v & 0xff);
    buf[2 * i + 1] = static_cast<uint8_t>((static_cast<uint16_t>(v) >> 8) & 0xff);
  }
  if (count && fwrite(buf.data(), 1, buf.size(), f) != buf.size()) {
    throw std::runtime_error("short write (disk full?)");
  }
  frames_ += count;
}

void WavStreamWriter::close() {
  if (!file_) return;
  FILE* f = static_cast<FILE*>(file_);
  file_ = nullptr;
  std::fseek(f, 0, SEEK_SET);
  writeHeader(f, sampleRate_, static_cast<uint32_t>(frames_ * 2));
  std::fclose(f);
}

// ---------------------------------------------------------------- resampler

StreamResampler::StreamResampler(int srcRate, int dstRate, int halfTaps)
    : src_(srcRate), dst_(dstRate), halfTaps_(halfTaps) {
  if (srcRate <= 0 || dstRate <= 0) throw std::invalid_argument("sample rates must be positive");
  step_ = static_cast<double>(srcRate) / dstRate;
  // Keep a small guard band below the lower Nyquist to limit aliasing.
  cutoff_ = std::min(1.0, static_cast<double>(dstRate) / srcRate) * 0.94;
  span_ = static_cast<int>(std::ceil(halfTaps_ / std::min(1.0, cutoff_)));
  history_.assign(static_cast<size_t>(span_), 0.0f);
  pos_ = span_;
  constexpr int kRes = 4096;
  kaiser_.resize(kRes + 2);
  const double norm = besselI0(kaiserBeta_);
  for (int i = 0; i <= kRes; ++i) {
    const double x = static_cast<double>(i) / kRes;
    kaiser_[static_cast<size_t>(i)] = static_cast<float>(besselI0(kaiserBeta_ * std::sqrt(std::max(0.0, 1.0 - x * x))) / norm);
  }
  kaiser_[kRes + 1] = 0.0f;
}

float StreamResampler::sampleAt(double pos) const {
  const long i0 = static_cast<long>(std::floor(pos));
  const double res = static_cast<double>(kaiser_.size() - 2);
  double acc = 0.0;
  for (long k = i0 - span_ + 1; k <= i0 + span_; ++k) {
    if (k < 0 || k >= static_cast<long>(history_.size())) continue;
    const double t = pos - static_cast<double>(k);
    const double x = t / span_;
    if (x <= -1.0 || x >= 1.0) continue;
    const double fi = std::fabs(x) * res;
    const size_t idx = static_cast<size_t>(fi);
    const double frac = fi - static_cast<double>(idx);
    const double w = kaiser_[idx] * (1.0 - frac) + kaiser_[idx + 1] * frac;
    const double a = kPi * cutoff_ * t;
    const double sinc = std::fabs(a) < 1e-9 ? 1.0 : std::sin(a) / a;
    acc += history_[static_cast<size_t>(k)] * cutoff_ * sinc * w;
  }
  return static_cast<float>(acc);
}

void StreamResampler::process(const float* in, size_t count, std::vector<float>& out) {
  if (src_ == dst_) {
    out.insert(out.end(), in, in + count);
    return;
  }
  history_.insert(history_.end(), in, in + count);
  while (static_cast<long>(std::floor(pos_)) + span_ < static_cast<long>(history_.size())) {
    out.push_back(sampleAt(pos_));
    pos_ += step_;
  }
  const long drop = static_cast<long>(std::floor(pos_)) - span_;
  if (drop > 0) {
    history_.erase(history_.begin(), history_.begin() + drop);
    pos_ -= static_cast<double>(drop);
  }
}

void StreamResampler::flush(std::vector<float>& out) {
  if (src_ == dst_) return;
  // Samples of real input still ahead of pos_ (history_ holds span_ of
  // look-behind; everything from pos_ onward is unconsumed real input).
  const double remainingIn = static_cast<double>(history_.size()) - pos_;
  const long remainingOut = remainingIn > 0 ? static_cast<long>(std::ceil(remainingIn / step_)) : 0;
  history_.insert(history_.end(), static_cast<size_t>(span_ + 1), 0.0f);
  for (long i = 0; i < remainingOut; ++i) {
    out.push_back(sampleAt(pos_));
    pos_ += step_;
  }
  history_.assign(static_cast<size_t>(span_), 0.0f);
  pos_ = span_;
}

std::vector<float> resample(const float* in, size_t count, int srcRate, int dstRate) {
  std::vector<float> out;
  if (srcRate == dstRate) return std::vector<float>(in, in + count);
  out.reserve(static_cast<size_t>(count * (static_cast<double>(dstRate) / srcRate)) + 8);
  StreamResampler r(srcRate, dstRate);
  r.process(in, count, out);
  r.flush(out);
  return out;
}

float rms(const float* x, size_t n) {
  if (!n) return 0.0f;
  double s = 0.0;
  for (size_t i = 0; i < n; ++i) s += static_cast<double>(x[i]) * x[i];
  return static_cast<float>(std::sqrt(s / n));
}

float peak(const float* x, size_t n) {
  float p = 0.0f;
  for (size_t i = 0; i < n; ++i) p = std::max(p, std::fabs(x[i]));
  return p;
}

void clipInPlace(float* x, size_t n, float limit) {
  for (size_t i = 0; i < n; ++i) x[i] = std::max(-limit, std::min(limit, x[i]));
}

}  // namespace mobigpt::rvc
