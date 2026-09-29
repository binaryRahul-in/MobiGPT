// MobiGPT Voice — DSP front-end shared by the neural pitch trackers.
//
// RMVPE and FCPE both consume a 128-bin log-mel spectrogram of 16 kHz audio
// (n_fft = win = 1024, hop = 160). They differ in mel scale / fmin / padding,
// so everything here is parameterised and checked against librosa in tests.
#pragma once

#include <complex>
#include <cstddef>
#include <vector>

namespace mobigpt::rvc {

constexpr double kPi = 3.14159265358979323846;

// In-place iterative radix-2 FFT. `n` must be a power of two.
void fft(std::complex<float>* data, size_t n, bool inverse = false);
void fft(std::complex<double>* data, size_t n, bool inverse = false);

// Periodic Hann window (torch.hann_window default).
std::vector<float> hannWindow(int length);

enum class MelScale { Slaney, Htk };

// librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax, htk, norm="slaney").
// Returns row-major [n_mels][n_fft/2 + 1].
std::vector<float> melFilterbank(int sampleRate, int nFft, int nMels, double fmin, double fmax,
                                 MelScale scale);

enum class StftPadding {
  // torch.stft(center=True, pad_mode="reflect") — used by RMVPE.
  CenterReflect,
  // FCPE: reflect-pad (win-hop)/2 left and enough right, center=False.
  FcpeReflect,
};

struct MelConfig {
  int sampleRate = 16000;
  int nFft = 1024;
  int winLength = 1024;
  int hop = 160;
  int nMels = 128;
  double fmin = 30.0;
  double fmax = 8000.0;
  MelScale scale = MelScale::Htk;
  StftPadding padding = StftPadding::CenterReflect;
  float clampMin = 1e-5f;
  float magnitudeEps = 0.0f;  // added under the sqrt (FCPE uses 1e-9)
};

MelConfig rmvpeMelConfig();
MelConfig fcpeMelConfig();

class MelExtractor {
 public:
  explicit MelExtractor(const MelConfig& cfg);
  // Log-mel, row-major [n_mels][frames] (a.k.a. "channels first").
  std::vector<float> compute(const float* audio, size_t count, int* framesOut) const;
  const MelConfig& config() const { return cfg_; }

 private:
  MelConfig cfg_;
  std::vector<float> window_;
  std::vector<float> fb_;
};

// Reflect-pad like numpy.pad(mode="reflect") (edge sample not repeated).
std::vector<float> reflectPad(const float* x, size_t n, size_t left, size_t right);

}  // namespace mobigpt::rvc
