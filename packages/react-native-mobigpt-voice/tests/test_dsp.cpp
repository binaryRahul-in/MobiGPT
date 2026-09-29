#include <cmath>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <vector>

#include "Audio.h"
#include "Dsp.h"
#include "OrtSession.h"
#include "testing.h"

using namespace mobigpt::rvc;

namespace {

std::vector<float> sine(double hz, double seconds, int sr, double amp = 0.5) {
  std::vector<float> x(static_cast<size_t>(seconds * sr));
  for (size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>(amp * std::sin(2 * kPi * hz * i / sr));
  return x;
}

std::vector<float> readRef(const std::string& name, int* rows, int* cols) {
  std::ifstream f(mt::fixture(name), std::ios::binary);
  int32_t dims[2];
  f.read(reinterpret_cast<char*>(dims), sizeof(dims));
  *rows = dims[0];
  *cols = dims[1];
  std::vector<float> v(static_cast<size_t>(dims[0]) * dims[1]);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * 4));
  return v;
}

double dominantHz(const std::vector<float>& x, int sr) {
  // Zero-crossing estimate on the steady middle part.
  size_t a = x.size() / 4, b = 3 * x.size() / 4, zc = 0;
  for (size_t i = a + 1; i < b; ++i) {
    if ((x[i - 1] < 0) != (x[i] < 0)) ++zc;
  }
  return zc / 2.0 / ((b - a) / static_cast<double>(sr));
}

}  // namespace

TEST(fft_matches_dft) {
  const size_t n = 64;
  std::vector<std::complex<float>> a(n);
  for (size_t i = 0; i < n; ++i) a[i] = {static_cast<float>(std::sin(0.3 * i) + 0.1 * i), 0.0f};
  auto ref = a;
  fft(a.data(), n);
  for (size_t k = 0; k < n; ++k) {
    std::complex<double> s = 0;
    for (size_t t = 0; t < n; ++t) s += std::complex<double>(ref[t]) * std::polar(1.0, -2 * kPi * k * t / n);
    REQUIRE_NEAR(a[k].real(), s.real(), 1e-3);
    REQUIRE_NEAR(a[k].imag(), s.imag(), 1e-3);
  }
  fft(a.data(), n, true);
  for (size_t i = 0; i < n; ++i) REQUIRE_NEAR(a[i].real(), ref[i].real(), 1e-4);
}

TEST(mel_filterbanks_match_librosa) {
  SKIP_UNLESS_FIXTURE("fb_htk.bin");
  int r = 0, c = 0;
  auto htk = readRef("fb_htk.bin", &r, &c);
  auto mine = melFilterbank(16000, 1024, 128, 30, 8000, MelScale::Htk);
  REQUIRE(r == 128 && c == 513);
  for (size_t i = 0; i < htk.size(); ++i) REQUIRE_NEAR(mine[i], htk[i], 1e-6);
  auto sl = readRef("fb_slaney.bin", &r, &c);
  auto mine2 = melFilterbank(16000, 1024, 128, 0, 8000, MelScale::Slaney);
  for (size_t i = 0; i < sl.size(); ++i) REQUIRE_NEAR(mine2[i], sl[i], 1e-6);
}

TEST(log_mel_matches_librosa_for_rmvpe_and_fcpe) {
  SKIP_UNLESS_FIXTURE("mel_rmvpe.bin");
  std::ifstream f(mt::fixture("mel_input.f32"), std::ios::binary);
  std::vector<float> y(8000);
  f.read(reinterpret_cast<char*>(y.data()), 8000 * 4);
  for (auto [file, cfg] : {std::make_pair("mel_rmvpe.bin", rmvpeMelConfig()), std::make_pair("mel_fcpe.bin", fcpeMelConfig())}) {
    int r = 0, c = 0;
    auto ref = readRef(file, &r, &c);
    int frames = 0;
    auto mel = MelExtractor(cfg).compute(y.data(), y.size(), &frames);
    REQUIRE(r == 128);
    REQUIRE(frames == c);
    double maxErr = 0, sumErr = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
      const double e = std::fabs(static_cast<double>(mel[i]) - ref[i]);
      maxErr = std::max(maxErr, e);
      sumErr += e;
    }
    const double meanErr = sumErr / ref.size();
    std::printf("    %s log-mel |err| max %.2e mean %.2e (frames=%d)\n", file, maxErr, meanErr, frames);
    // float32 FFT vs librosa's float64: a few 1e-3 in log domain (<0.5 %% in magnitude).
    REQUIRE(maxErr < 1e-2);
    REQUIRE(meanErr < 5e-4);
  }
}

TEST(reflect_pad_matches_numpy) {
  const float x[] = {1, 2, 3, 4};
  auto p = reflectPad(x, 4, 3, 2);
  const float expect[] = {4, 3, 2, 1, 2, 3, 4, 3, 2};
  REQUIRE(p.size() == 9);
  for (size_t i = 0; i < 9; ++i) REQUIRE(p[i] == expect[i]);
}

TEST(resampler_preserves_frequency_and_length) {
  for (auto [src, dst] : {std::make_pair(44100, 16000), std::make_pair(48000, 16000), std::make_pair(16000, 40000),
                          std::make_pair(22050, 16000)}) {
    auto x = sine(440.0, 1.0, src);
    auto y = resample(x.data(), x.size(), src, dst);
    REQUIRE_NEAR(static_cast<double>(y.size()), static_cast<double>(dst), 2.0);
    REQUIRE_NEAR(dominantHz(y, dst), 440.0, 3.0);
    REQUIRE_NEAR(rms(y.data() + dst / 4, dst / 2), 0.5 / std::sqrt(2.0), 0.01);
  }
}

TEST(resampler_attenuates_aliasing) {
  // 6 kHz at 44.1k is kept; 9 kHz must not fold back into the 16k band.
  auto x = sine(6000.0, 0.5, 44100);
  auto y16 = resample(x.data(), x.size(), 44100, 16000);
  REQUIRE(rms(y16.data() + 1000, y16.size() - 2000) > 0.3);  // passband kept
  auto z = sine(9000.0, 0.5, 44100);
  auto z16 = resample(z.data(), z.size(), 44100, 16000);
  REQUIRE(rms(z16.data() + 1000, z16.size() - 2000) < 0.02);  // 9 kHz removed
}

TEST(stream_resampler_equals_offline) {
  auto x = sine(300.0, 0.7, 48000);
  auto off = resample(x.data(), x.size(), 48000, 16000);
  StreamResampler rs(48000, 16000);
  std::vector<float> on;
  for (size_t i = 0; i < x.size(); i += 777) rs.process(x.data() + i, std::min<size_t>(777, x.size() - i), on);
  rs.flush(on);
  REQUIRE(on.size() == off.size());
  for (size_t i = 0; i < on.size(); ++i) REQUIRE_NEAR(on[i], off[i], 1e-5);
}

TEST(wav_roundtrip_pcm16) {
  auto x = sine(220.0, 0.25, 22050, 0.8);
  const std::string p = "/tmp/mobigpt_test_rt.wav";
  writeWavPcm16(p, x.data(), x.size(), 22050);
  auto b = readWav(p);
  REQUIRE(b.sampleRate == 22050);
  REQUIRE(b.samples.size() == x.size());
  for (size_t i = 0; i < x.size(); ++i) REQUIRE_NEAR(b.samples[i], x[i], 1.0 / 16000);
  std::remove(p.c_str());
}

TEST(wav_parses_stereo_float_and_24bit) {
  // Build a stereo float32 WAV by hand.
  std::vector<uint8_t> w;
  auto put32 = [&](uint32_t v) { for (int i = 0; i < 4; ++i) w.push_back(static_cast<uint8_t>(v >> (8 * i))); };
  auto put16 = [&](uint16_t v) { w.push_back(static_cast<uint8_t>(v)); w.push_back(static_cast<uint8_t>(v >> 8)); };
  auto tag = [&](const char* s) { w.insert(w.end(), s, s + 4); };
  const float frames[][2] = {{0.5f, -0.5f}, {1.0f, 0.0f}};
  tag("RIFF"); put32(36 + 16); tag("WAVE"); tag("fmt "); put32(16); put16(3); put16(2); put32(8000); put32(8000 * 8);
  put16(8); put16(32); tag("data"); put32(16);
  for (auto& fr : frames) for (float s : fr) { uint32_t u; std::memcpy(&u, &s, 4); put32(u); }
  auto b = parseWav(w.data(), w.size());
  REQUIRE(b.sampleRate == 8000 && b.samples.size() == 2);
  REQUIRE_NEAR(b.samples[0], 0.0, 1e-6);
  REQUIRE_NEAR(b.samples[1], 0.5, 1e-6);
  REQUIRE_THROWS(parseWav(w.data(), 10));
}

TEST(half_float_roundtrip) {
  for (float v : {0.0f, 1.0f, -2.5f, 0.000123f, 65504.0f, 3.14159f}) {
    REQUIRE_NEAR(halfToFloat(floatToHalf(v)), v, std::fabs(v) * 1e-3 + 1e-7);
  }
}
