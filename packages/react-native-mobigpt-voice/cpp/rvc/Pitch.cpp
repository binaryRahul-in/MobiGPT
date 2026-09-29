#include "Pitch.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

#include "Dsp.h"
#include "world/dio.h"
#include "world/harvest.h"
#include "world/stonemask.h"

namespace mobigpt::rvc {

PitchMethod pitchMethodFromString(const std::string& s) {
  if (s == "rmvpe") return PitchMethod::Rmvpe;
  if (s == "fcpe") return PitchMethod::Fcpe;
  if (s == "harvest") return PitchMethod::Harvest;
  if (s == "dio") return PitchMethod::Dio;
  if (s == "pm") return PitchMethod::Pm;
  throw std::invalid_argument("unknown pitch method: " + s);
}

const char* pitchMethodName(PitchMethod m) {
  switch (m) {
    case PitchMethod::Rmvpe: return "rmvpe";
    case PitchMethod::Fcpe: return "fcpe";
    case PitchMethod::Harvest: return "harvest";
    case PitchMethod::Dio: return "dio";
    case PitchMethod::Pm: return "pm";
  }
  return "unknown";
}

bool pitchMethodNeedsModel(PitchMethod m) { return m == PitchMethod::Rmvpe || m == PitchMethod::Fcpe; }

namespace {

constexpr int kSr = 16000;
constexpr int kHop = 160;

std::vector<float> fitFrames(std::vector<float> f0, size_t frames) {
  if (f0.size() > frames) {
    f0.resize(frames);
  } else if (f0.size() < frames) {
    const float last = f0.empty() ? 0.0f : f0.back();
    f0.resize(frames, last);
  }
  return f0;
}

void median3(std::vector<float>& f0) {
  if (f0.size() < 3) return;
  std::vector<float> out(f0);
  for (size_t i = 1; i + 1 < f0.size(); ++i) {
    float a = f0[i - 1], b = f0[i], c = f0[i + 1];
    out[i] = std::max(std::min(a, b), std::min(std::max(a, b), c));
  }
  f0.swap(out);
}

void clampRange(std::vector<float>& f0, double lo, double hi) {
  for (auto& v : f0) {
    if (v < lo * 0.9 || v > hi * 1.1 || !std::isfinite(v)) v = 0.0f;
  }
}

// --------------------------------------------------------------------- RMVPE
class RmvpeExtractor final : public PitchExtractor {
 public:
  RmvpeExtractor(const std::string& path, const SessionConfig& cfg, const PitchOptions& opts)
      : model_(path, cfg), mel_(rmvpeMelConfig()), opts_(opts) {
    waveformInput_ = model_.hasInput("waveform");
    if (!waveformInput_) {
      const auto& in = model_.inputs().at(0);
      if (in.shape.size() != 3) throw std::runtime_error("RMVPE: expected rank-3 mel input");
      timeMajor_ = in.shape[2] == 128 && in.shape[1] != 128;
    }
  }

  const char* name() const override { return "rmvpe"; }

  std::vector<float> extract(const float* audio, size_t count, size_t frames) override {
    const float thr = opts_.threshold >= 0 ? opts_.threshold : 0.03f;
    std::vector<float> f0;
    if (waveformInput_) {
      // w-okada export: mel + decode are inside the graph, output is Hz.
      std::vector<float> wav(audio, audio + count);
      float thrv = thr;
      std::vector<Ort::Value> ins;
      std::vector<const char*> names{"waveform"};
      ins.push_back(borrowTensor(wav.data(), wav.size(), {1, static_cast<int64_t>(wav.size())}));
      if (model_.hasInput("threshold")) {
        names.push_back("threshold");
        ins.push_back(borrowTensor(&thrv, 1, {1}));
      }
      const char* outName = model_.outputs().at(0).name.c_str();
      auto outs = model_.run(names, ins, {outName});
      f0 = toFloatVector(outs[0]);
    } else {
      int t = 0;
      std::vector<float> mel = mel_.compute(audio, count, &t);  // [128][t]
      const int tp = 32 * ((t - 1) / 32 + 1);
      std::vector<float> in(static_cast<size_t>(128) * tp, 0.0f);
      for (int m = 0; m < 128; ++m) {
        for (int i = 0; i < tp; ++i) {
          // Reflect pad the tail (matches RVC for n_pad < n_frames).
          int src = i < t ? i : std::max(0, 2 * (t - 1) - i);
          const float v = mel[static_cast<size_t>(m) * t + src];
          if (timeMajor_) in[static_cast<size_t>(i) * 128 + m] = v;
          else in[static_cast<size_t>(m) * tp + i] = v;
        }
      }
      std::vector<int64_t> shape = timeMajor_ ? std::vector<int64_t>{1, tp, 128} : std::vector<int64_t>{1, 128, tp};
      std::vector<Ort::Value> ins;
      ins.push_back(borrowTensor(in.data(), in.size(), shape));
      const std::string inName = model_.inputs().at(0).name;
      std::vector<const char*> names{inName.c_str()};
      float thrv = thr;
      if (model_.hasInput("threshold")) {
        ins.push_back(borrowTensor(&thrv, 1, {1}));
        names.push_back("threshold");
      }
      const char* outName = model_.outputs().at(0).name.c_str();
      auto outs = model_.run(names, ins, {outName});
      std::vector<int64_t> oshape;
      std::vector<float> out = toFloatVector(outs[0], &oshape);
      if (!oshape.empty() && oshape.back() == 360) {
        f0 = decodeRmvpeSalience(out.data(), std::min<size_t>(static_cast<size_t>(t), out.size() / 360), thr);
      } else {
        f0 = std::move(out);  // model already decodes to Hz
        f0.resize(std::min<size_t>(f0.size(), static_cast<size_t>(t)));
      }
    }
    clampRange(f0, opts_.f0Min, opts_.f0Max);
    return fitFrames(std::move(f0), frames);
  }

 private:
  OnnxModel model_;
  MelExtractor mel_;
  PitchOptions opts_;
  bool waveformInput_ = false;
  bool timeMajor_ = false;
};

// ---------------------------------------------------------------------- FCPE
class FcpeExtractor final : public PitchExtractor {
 public:
  FcpeExtractor(const std::string& path, const SessionConfig& cfg, const PitchOptions& opts)
      : model_(path, cfg), mel_(fcpeMelConfig()), opts_(opts) {
    const auto& in = model_.inputs().at(0);
    waveformInput_ = in.shape.size() == 2 || in.name == "waveform" || in.name == "wav" || in.name == "audio";
    if (!waveformInput_) channelsFirst_ = in.shape.size() == 3 && in.shape[1] == 128 && in.shape[2] != 128;
  }

  const char* name() const override { return "fcpe"; }

  std::vector<float> extract(const float* audio, size_t count, size_t frames) override {
    const float thr = opts_.threshold >= 0 ? opts_.threshold : 0.006f;
    std::vector<Ort::Value> ins;
    std::vector<const char*> names;
    std::vector<float> buf;
    float thrv = thr;
    const std::string inName = model_.inputs().at(0).name;
    int t = 0;
    if (waveformInput_) {
      buf.assign(audio, audio + count);
      ins.push_back(borrowTensor(buf.data(), buf.size(), {1, static_cast<int64_t>(buf.size())}));
      names.push_back(inName.c_str());
      if (model_.hasInput("threshold")) {
        ins.push_back(borrowTensor(&thrv, 1, {1}));
        names.push_back("threshold");
      }
    } else {
      std::vector<float> mel = mel_.compute(audio, count, &t);  // [128][t]
      buf.resize(mel.size());
      if (channelsFirst_) {
        buf = mel;
      } else {
        for (int m = 0; m < 128; ++m)
          for (int i = 0; i < t; ++i) buf[static_cast<size_t>(i) * 128 + m] = mel[static_cast<size_t>(m) * t + i];
      }
      std::vector<int64_t> shape = channelsFirst_ ? std::vector<int64_t>{1, 128, t} : std::vector<int64_t>{1, t, 128};
      ins.push_back(borrowTensor(buf.data(), buf.size(), shape));
      names.push_back(inName.c_str());
      // Exports that decode inside the graph (e.g. niobures/FCPE) take the voicing
      // threshold as a second input and return Hz.
      if (model_.hasInput("threshold")) {
        ins.push_back(borrowTensor(&thrv, 1, {1}));
        names.push_back("threshold");
      }
    }
    const char* outName = model_.outputs().at(0).name.c_str();
    auto outs = model_.run(names, ins, {outName});
    std::vector<int64_t> oshape;
    std::vector<float> out = toFloatVector(outs[0], &oshape);
    std::vector<float> f0;
    if (!oshape.empty() && oshape.back() == 360) {
      f0 = decodeFcpeLatent(out.data(), out.size() / 360, thr);
    } else {
      f0 = std::move(out);
    }
    clampRange(f0, opts_.f0Min, opts_.f0Max);
    return fitFrames(std::move(f0), frames);
  }

 private:
  OnnxModel model_;
  MelExtractor mel_;
  PitchOptions opts_;
  bool waveformInput_ = false;
  bool channelsFirst_ = false;
};

// --------------------------------------------------------------- WORLD (DSP)
class WorldExtractor final : public PitchExtractor {
 public:
  WorldExtractor(bool harvest, const PitchOptions& opts) : harvest_(harvest), opts_(opts) {}
  const char* name() const override { return harvest_ ? "harvest" : "dio"; }

  std::vector<float> extract(const float* audio, size_t count, size_t frames) override {
    if (count == 0) return std::vector<float>(frames, 0.0f);
    std::vector<double> x(audio, audio + count);
    const int len = static_cast<int>(count);
    std::vector<double> f0, tpos;
    if (harvest_) {
      HarvestOption o;
      InitializeHarvestOption(&o);
      o.f0_floor = opts_.f0Min;
      o.f0_ceil = opts_.f0Max;
      o.frame_period = 10.0;
      const int n = GetSamplesForHarvest(kSr, len, o.frame_period);
      f0.resize(static_cast<size_t>(n));
      tpos.resize(static_cast<size_t>(n));
      Harvest(x.data(), len, kSr, &o, tpos.data(), f0.data());
    } else {
      DioOption o;
      InitializeDioOption(&o);
      o.f0_floor = opts_.f0Min;
      o.f0_ceil = opts_.f0Max;
      o.frame_period = 10.0;
      o.speed = 1;
      o.allowed_range = 0.1;
      const int n = GetSamplesForDIO(kSr, len, o.frame_period);
      f0.resize(static_cast<size_t>(n));
      tpos.resize(static_cast<size_t>(n));
      Dio(x.data(), len, kSr, &o, tpos.data(), f0.data());
      std::vector<double> refined(f0.size());
      StoneMask(x.data(), len, kSr, tpos.data(), f0.data(), n, refined.data());
      f0.swap(refined);
    }
    std::vector<float> out(f0.begin(), f0.end());
    median3(out);
    clampRange(out, opts_.f0Min, opts_.f0Max);
    return fitFrames(std::move(out), frames);
  }

 private:
  bool harvest_;
  PitchOptions opts_;
};

class PmExtractor final : public PitchExtractor {
 public:
  explicit PmExtractor(const PitchOptions& o) : opts_(o) {}
  const char* name() const override { return "pm"; }
  std::vector<float> extract(const float* audio, size_t count, size_t frames) override {
    auto f0 = pmPitch(audio, count, frames, opts_.f0Min, opts_.f0Max);
    median3(f0);
    return f0;
  }

 private:
  PitchOptions opts_;
};

}  // namespace

std::vector<float> decodeRmvpeSalience(const float* s, size_t frames, float threshold) {
  std::vector<float> f0(frames, 0.0f);
  for (size_t t = 0; t < frames; ++t) {
    const float* row = s + t * 360;
    int center = 0;
    float maxv = row[0];
    for (int i = 1; i < 360; ++i) {
      if (row[i] > maxv) {
        maxv = row[i];
        center = i;
      }
    }
    if (maxv <= threshold) continue;
    double num = 0.0, den = 0.0;
    for (int i = center - 4; i <= center + 4; ++i) {
      if (i < 0 || i >= 360) continue;  // zero-padded in the reference
      const double cents = 20.0 * i + 1997.3794084376191;
      num += row[i] * cents;
      den += row[i];
    }
    if (den <= 0.0) continue;
    const double hz = 10.0 * std::pow(2.0, (num / den) / 1200.0);
    f0[t] = static_cast<float>(hz);
  }
  return f0;
}

std::vector<float> decodeFcpeLatent(const float* y, size_t frames, float threshold) {
  const double c0 = 1200.0 * std::log2(32.70 / 10.0);
  const double c1 = 1200.0 * std::log2(1975.5 / 10.0);
  std::vector<float> f0(frames, 0.0f);
  for (size_t t = 0; t < frames; ++t) {
    const float* row = y + t * 360;
    int center = 0;
    float maxv = row[0];
    for (int i = 1; i < 360; ++i) {
      if (row[i] > maxv) {
        maxv = row[i];
        center = i;
      }
    }
    if (maxv <= threshold) continue;
    double num = 0.0, den = 0.0;
    for (int k = center - 4; k <= center + 4; ++k) {
      const int i = std::max(0, std::min(359, k));  // torch.gather clamps
      const double cents = c0 + (c1 - c0) * i / 359.0;
      num += row[i] * cents;
      den += row[i];
    }
    if (den <= 0.0) continue;
    f0[t] = static_cast<float>(10.0 * std::pow(2.0, (num / den) / 1200.0));
  }
  return f0;
}

std::vector<float> pmPitch(const float* x, size_t count, size_t frames, double f0Min, double f0Max) {
  std::vector<float> f0(frames, 0.0f);
  if (count == 0) return f0;
  const int win = static_cast<int>(std::ceil(3.0 * kSr / f0Min));  // 3 periods of the lowest pitch
  size_t nfft = 1;
  while (nfft < static_cast<size_t>(2 * win)) nfft <<= 1;
  const int lagMin = static_cast<int>(std::floor(kSr / f0Max));
  const int lagMax = std::min(win - 1, static_cast<int>(std::ceil(kSr / f0Min)));

  // Autocorrelation of the analysis window, used to undo its taper (Boersma 1993).
  const std::vector<float> hann = hannWindow(win);
  std::vector<std::complex<float>> buf(nfft);
  auto autocorr = [&](std::vector<double>& r) {
    fft(buf.data(), nfft);
    for (auto& z : buf) z = std::complex<float>(std::norm(z), 0.0f);
    fft(buf.data(), nfft, true);
    r.resize(static_cast<size_t>(win));
    for (int i = 0; i < win; ++i) r[static_cast<size_t>(i)] = buf[static_cast<size_t>(i)].real();
  };
  std::vector<double> rw;
  std::fill(buf.begin(), buf.end(), std::complex<float>(0.0f));
  for (int i = 0; i < win; ++i) buf[static_cast<size_t>(i)] = hann[static_cast<size_t>(i)];
  autocorr(rw);

  const float globalPeak = std::max(1e-6f, [&] {
    float p = 0.0f;
    for (size_t i = 0; i < count; ++i) p = std::max(p, std::fabs(x[i]));
    return p;
  }());

  std::vector<double> r;
  for (size_t t = 0; t < frames; ++t) {
    const long center = static_cast<long>(t) * kHop;
    const long start = center - win / 2;
    double mean = 0.0, localPeak = 0.0;
    int valid = 0;
    for (int i = 0; i < win; ++i) {
      const long k = start + i;
      if (k >= 0 && k < static_cast<long>(count)) {
        mean += x[k];
        ++valid;
      }
    }
    if (valid < win / 2) continue;
    mean /= valid;
    std::fill(buf.begin(), buf.end(), std::complex<float>(0.0f));
    for (int i = 0; i < win; ++i) {
      const long k = start + i;
      const double v = (k >= 0 && k < static_cast<long>(count)) ? x[k] - mean : 0.0;
      localPeak = std::max(localPeak, std::fabs(v));
      buf[static_cast<size_t>(i)] = static_cast<float>(v * hann[static_cast<size_t>(i)]);
    }
    if (localPeak < 0.03 * globalPeak) continue;  // silence gate
    autocorr(r);
    if (r[0] <= 0.0) continue;
    auto nv = [&](int l) { return (r[static_cast<size_t>(l)] / r[0]) / (rw[static_cast<size_t>(l)] / rw[0]); };
    double bestV = 0.0;
    for (int lag = lagMin + 1; lag < lagMax; ++lag) bestV = std::max(bestV, nv(lag));
    // Voicing threshold loosely follows Praat's default (0.45) for speech.
    if (bestV < 0.45) continue;
    // Take the *shortest* lag whose local peak is within 10 % of the global
    // maximum: longer lags are period multiples (octave-down errors).
    int best = -1;
    for (int lag = lagMin + 1; lag < lagMax; ++lag) {
      const double v = nv(lag);
      if (v >= 0.9 * bestV && v >= nv(lag - 1) && v >= nv(lag + 1)) {
        best = lag;
        break;
      }
    }
    if (best < 0) continue;
    const double a = nv(best - 1), b = nv(best), c = nv(best + 1);
    const double denom = a - 2 * b + c;
    const double shift = std::fabs(denom) > 1e-12 ? 0.5 * (a - c) / denom : 0.0;
    const double period = best + std::max(-0.5, std::min(0.5, shift));
    const double hz = kSr / period;
    if (hz >= f0Min && hz <= f0Max) f0[t] = static_cast<float>(hz);
  }
  return f0;
}

std::unique_ptr<PitchExtractor> createPitchExtractor(PitchMethod method, const std::string& modelPath,
                                                     const SessionConfig& session, const PitchOptions& opts) {
  switch (method) {
    case PitchMethod::Rmvpe:
      if (modelPath.empty()) throw std::invalid_argument("RMVPE needs a model file");
      return std::make_unique<RmvpeExtractor>(modelPath, session, opts);
    case PitchMethod::Fcpe:
      if (modelPath.empty()) throw std::invalid_argument("FCPE needs a model file");
      return std::make_unique<FcpeExtractor>(modelPath, session, opts);
    case PitchMethod::Harvest:
      return std::make_unique<WorldExtractor>(true, opts);
    case PitchMethod::Dio:
      return std::make_unique<WorldExtractor>(false, opts);
    case PitchMethod::Pm:
      return std::make_unique<PmExtractor>(opts);
  }
  throw std::invalid_argument("bad pitch method");
}

}  // namespace mobigpt::rvc
