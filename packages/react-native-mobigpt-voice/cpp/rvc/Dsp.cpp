#include "Dsp.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mobigpt::rvc {

namespace {
template <typename T>
void fftImpl(std::complex<T>* a, size_t n, bool inverse) {
  if (n == 0 || (n & (n - 1)) != 0) throw std::invalid_argument("fft size must be a power of two");
  for (size_t i = 1, j = 0; i < n; ++i) {
    size_t bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(a[i], a[j]);
  }
  for (size_t len = 2; len <= n; len <<= 1) {
    const double ang = 2.0 * kPi / static_cast<double>(len) * (inverse ? 1.0 : -1.0);
    const std::complex<double> wlen(std::cos(ang), std::sin(ang));
    for (size_t i = 0; i < n; i += len) {
      std::complex<double> w(1.0, 0.0);
      for (size_t k = 0; k < len / 2; ++k) {
        const std::complex<T> u = a[i + k];
        const std::complex<T> v = a[i + k + len / 2] * std::complex<T>(w);
        a[i + k] = u + v;
        a[i + k + len / 2] = u - v;
        w *= wlen;
      }
    }
  }
  if (inverse) {
    for (size_t i = 0; i < n; ++i) a[i] /= static_cast<T>(n);
  }
}
}  // namespace

void fft(std::complex<float>* a, size_t n, bool inverse) { fftImpl(a, n, inverse); }
void fft(std::complex<double>* a, size_t n, bool inverse) { fftImpl(a, n, inverse); }

std::vector<float> hannWindow(int length) {
  std::vector<float> w(static_cast<size_t>(length));
  for (int i = 0; i < length; ++i) {
    w[static_cast<size_t>(i)] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPi * i / length));
  }
  return w;
}

namespace {

double hzToMel(double f, MelScale s) {
  if (s == MelScale::Htk) return 2595.0 * std::log10(1.0 + f / 700.0);
  const double fSp = 200.0 / 3.0;
  const double minLogHz = 1000.0;
  const double minLogMel = minLogHz / fSp;
  const double logstep = std::log(6.4) / 27.0;
  if (f >= minLogHz) return minLogMel + std::log(f / minLogHz) / logstep;
  return f / fSp;
}

double melToHz(double m, MelScale s) {
  if (s == MelScale::Htk) return 700.0 * (std::pow(10.0, m / 2595.0) - 1.0);
  const double fSp = 200.0 / 3.0;
  const double minLogHz = 1000.0;
  const double minLogMel = minLogHz / fSp;
  const double logstep = std::log(6.4) / 27.0;
  if (m >= minLogMel) return minLogHz * std::exp(logstep * (m - minLogMel));
  return fSp * m;
}

}  // namespace

std::vector<float> melFilterbank(int sampleRate, int nFft, int nMels, double fmin, double fmax,
                                 MelScale scale) {
  const int nBins = nFft / 2 + 1;
  std::vector<double> fftFreqs(static_cast<size_t>(nBins));
  for (int i = 0; i < nBins; ++i) fftFreqs[static_cast<size_t>(i)] = static_cast<double>(sampleRate) / 2.0 * i / (nBins - 1);

  const double melMin = hzToMel(fmin, scale);
  const double melMax = hzToMel(fmax, scale);
  std::vector<double> melF(static_cast<size_t>(nMels + 2));
  for (int i = 0; i < nMels + 2; ++i) {
    melF[static_cast<size_t>(i)] = melToHz(melMin + (melMax - melMin) * i / (nMels + 1), scale);
  }

  std::vector<float> w(static_cast<size_t>(nMels) * nBins, 0.0f);
  for (int m = 0; m < nMels; ++m) {
    const double lo = melF[static_cast<size_t>(m)];
    const double c = melF[static_cast<size_t>(m + 1)];
    const double hi = melF[static_cast<size_t>(m + 2)];
    const double enorm = 2.0 / (hi - lo);  // slaney area normalisation
    for (int k = 0; k < nBins; ++k) {
      const double f = fftFreqs[static_cast<size_t>(k)];
      const double lower = (f - lo) / (c - lo);
      const double upper = (hi - f) / (hi - c);
      const double v = std::max(0.0, std::min(lower, upper));
      w[static_cast<size_t>(m) * nBins + k] = static_cast<float>(v * enorm);
    }
  }
  return w;
}

std::vector<float> reflectPad(const float* x, size_t n, size_t left, size_t right) {
  std::vector<float> out(left + n + right, 0.0f);
  if (n == 0) return out;
  auto at = [&](long i) -> float {
    if (n == 1) return x[0];
    const long period = 2 * (static_cast<long>(n) - 1);
    long m = i % period;
    if (m < 0) m += period;
    if (m >= static_cast<long>(n)) m = period - m;
    return x[m];
  };
  for (size_t i = 0; i < out.size(); ++i) out[i] = at(static_cast<long>(i) - static_cast<long>(left));
  return out;
}

MelConfig rmvpeMelConfig() {
  MelConfig c;
  c.fmin = 30.0;
  c.fmax = 8000.0;
  c.scale = MelScale::Htk;
  c.padding = StftPadding::CenterReflect;
  return c;
}

MelConfig fcpeMelConfig() {
  MelConfig c;
  c.fmin = 0.0;
  c.fmax = 8000.0;
  c.scale = MelScale::Slaney;
  c.padding = StftPadding::FcpeReflect;
  c.magnitudeEps = 1e-9f;
  return c;
}

MelExtractor::MelExtractor(const MelConfig& cfg) : cfg_(cfg) {
  if ((cfg.nFft & (cfg.nFft - 1)) != 0) throw std::invalid_argument("nFft must be a power of two");
  window_ = hannWindow(cfg.winLength);
  fb_ = melFilterbank(cfg.sampleRate, cfg.nFft, cfg.nMels, cfg.fmin, cfg.fmax, cfg.scale);
}

std::vector<float> MelExtractor::compute(const float* audio, size_t count, int* framesOut) const {
  std::vector<float> padded;
  if (cfg_.padding == StftPadding::CenterReflect) {
    const size_t p = static_cast<size_t>(cfg_.nFft / 2);
    padded = count > p ? reflectPad(audio, count, p, p) : std::vector<float>(count + 2 * p, 0.0f);
    if (count <= p) std::copy(audio, audio + count, padded.begin() + static_cast<long>(p));
  } else {
    const long win = cfg_.winLength, hop = cfg_.hop;
    const long padL = (win - hop) / 2;
    const long padR = std::max((win - hop + 1) / 2, win - static_cast<long>(count) - padL);
    if (padR < static_cast<long>(count)) {
      padded = reflectPad(audio, count, static_cast<size_t>(padL), static_cast<size_t>(padR));
    } else {
      padded.assign(static_cast<size_t>(padL + static_cast<long>(count) + padR), 0.0f);
      std::copy(audio, audio + count, padded.begin() + padL);
    }
  }

  const int nFft = cfg_.nFft, win = cfg_.winLength, hop = cfg_.hop, nBins = nFft / 2 + 1;
  if (static_cast<long>(padded.size()) < nFft) padded.resize(static_cast<size_t>(nFft), 0.0f);
  const int frames = 1 + static_cast<int>((padded.size() - static_cast<size_t>(nFft)) / static_cast<size_t>(hop));
  const int winOffset = (nFft - win) / 2;  // torch centers a shorter window inside n_fft

  std::vector<float> mag(static_cast<size_t>(nBins));
  std::vector<std::complex<double>> buf(static_cast<size_t>(nFft));
  std::vector<float> mel(static_cast<size_t>(cfg_.nMels) * frames);
  for (int t = 0; t < frames; ++t) {
    const float* frame = padded.data() + static_cast<size_t>(t) * hop;
    std::fill(buf.begin(), buf.end(), std::complex<double>(0.0, 0.0));
    for (int i = 0; i < win; ++i) buf[static_cast<size_t>(i + winOffset)] = static_cast<double>(frame[i + winOffset]) * window_[static_cast<size_t>(i)];
    fft(buf.data(), buf.size());
    for (int k = 0; k < nBins; ++k) {
      const auto& z = buf[static_cast<size_t>(k)];
      mag[static_cast<size_t>(k)] = static_cast<float>(std::sqrt(z.real() * z.real() + z.imag() * z.imag() + cfg_.magnitudeEps));
    }
    for (int m = 0; m < cfg_.nMels; ++m) {
      const float* row = fb_.data() + static_cast<size_t>(m) * nBins;
      double acc = 0.0;
      for (int k = 0; k < nBins; ++k) acc += static_cast<double>(row[k]) * mag[static_cast<size_t>(k)];
      mel[static_cast<size_t>(m) * frames + t] = std::log(std::max(static_cast<float>(acc), cfg_.clampMin));
    }
  }
  if (framesOut) *framesOut = frames;
  return mel;
}

}  // namespace mobigpt::rvc
