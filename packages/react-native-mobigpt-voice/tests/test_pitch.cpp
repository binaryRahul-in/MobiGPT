#include <algorithm>
#include <cmath>
#include <vector>

#include "Dsp.h"
#include "Pitch.h"
#include "testing.h"

using namespace mobigpt::rvc;

namespace {

// Harmonic "voice": 8 decaying partials, optional vibrato.
std::vector<float> voice(double hz, double seconds, double vibrato = 0.0) {
  const int sr = 16000;
  std::vector<float> x(static_cast<size_t>(seconds * sr));
  double ph = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    const double f = hz * (1.0 + vibrato * std::sin(2 * kPi * 5 * i / sr));
    ph += 2 * kPi * f / sr;
    double s = 0;
    for (int h = 1; h <= 8; ++h) s += std::sin(h * ph) / h;
    x[i] = static_cast<float>(0.25 * s);
  }
  return x;
}

double medianVoiced(const std::vector<float>& f0, double* voicedRatio = nullptr) {
  std::vector<float> v;
  for (float x : f0) if (x > 0) v.push_back(x);
  if (voicedRatio) *voicedRatio = f0.empty() ? 0 : static_cast<double>(v.size()) / f0.size();
  if (v.empty()) return 0;
  std::nth_element(v.begin(), v.begin() + static_cast<long>(v.size() / 2), v.end());
  return v[v.size() / 2];
}

}  // namespace

TEST(dsp_trackers_find_pitch) {
  PitchOptions o;
  for (double hz : {110.0, 220.0, 330.0}) {
    auto x = voice(hz, 1.0);
    const size_t frames = x.size() / 160;
    for (auto m : {PitchMethod::Dio, PitchMethod::Harvest, PitchMethod::Pm}) {
      auto ex = createPitchExtractor(m, "", SessionConfig{}, o);
      auto f0 = ex->extract(x.data(), x.size(), frames);
      REQUIRE(f0.size() == frames);
      double vr = 0;
      const double med = medianVoiced(f0, &vr);
      std::printf("    %-8s %.0f Hz -> median %.1f Hz, voiced %.0f%%\n", ex->name(), hz, med, vr * 100);
      REQUIRE(std::fabs(med - hz) / hz < 0.03);
      REQUIRE(vr > 0.8);
    }
  }
}

TEST(dsp_trackers_report_silence_as_unvoiced) {
  std::vector<float> x(16000, 0.0f);
  for (auto m : {PitchMethod::Dio, PitchMethod::Harvest, PitchMethod::Pm}) {
    auto ex = createPitchExtractor(m, "", SessionConfig{}, PitchOptions{});
    auto f0 = ex->extract(x.data(), x.size(), 100);
    double vr = 1;
    medianVoiced(f0, &vr);
    REQUIRE(vr < 0.05);
  }
}

TEST(rmvpe_decode_matches_reference_formula) {
  std::vector<float> sal(3 * 360, 0.0f);
  sal[0 * 360 + 100] = 0.9f;                                           // single peak
  sal[1 * 360 + 200] = 0.5f; sal[1 * 360 + 201] = 0.5f;                // between two bins
  sal[2 * 360 + 50] = 0.01f;                                           // below threshold
  auto f0 = decodeRmvpeSalience(sal.data(), 3);
  REQUIRE_NEAR(f0[0], 10.0 * std::pow(2.0, (20.0 * 100 + 1997.3794084376191) / 1200.0), 1e-3);
  REQUIRE_NEAR(f0[1], 10.0 * std::pow(2.0, (20.0 * 200.5 + 1997.3794084376191) / 1200.0), 1e-3);
  REQUIRE(f0[2] == 0.0f);
}

TEST(fcpe_decode_spans_expected_range) {
  std::vector<float> lat(2 * 360, 0.0f);
  lat[0] = 1.0f;          // lowest bin -> 32.70 Hz
  lat[360 + 359] = 1.0f;  // highest bin -> 1975.5 Hz
  auto f0 = decodeFcpeLatent(lat.data(), 2);
  REQUIRE_NEAR(f0[0], 32.70, 0.05);
  REQUIRE_NEAR(f0[1], 1975.5, 0.5);
}

TEST(neural_extractors_run_on_all_layouts) {
  SKIP_UNLESS_FIXTURE("rmvpe_mel.onnx");
  auto x = voice(200, 1.3);
  const size_t frames = x.size() / 160;
  SessionConfig sc;
  PitchOptions o;
  o.f0Min = 30;
  o.f0Max = 2000;
  for (auto [m, file] : {std::make_pair(PitchMethod::Rmvpe, "rmvpe_mel.onnx"), std::make_pair(PitchMethod::Rmvpe, "rmvpe_wave.onnx"),
                         std::make_pair(PitchMethod::Fcpe, "fcpe.onnx")}) {
    auto ex = createPitchExtractor(m, mt::fixture(file), sc, o);
    auto f0 = ex->extract(x.data(), x.size(), frames);
    REQUIRE(f0.size() == frames);
    for (float v : f0) REQUIRE(std::isfinite(v) && v >= 0);
    if (std::string(file) == "rmvpe_wave.onnx") REQUIRE_NEAR(medianVoiced(f0), 220.0, 0.01);
  }
}

TEST(missing_model_is_an_error) {
  REQUIRE_THROWS(createPitchExtractor(PitchMethod::Rmvpe, "", SessionConfig{}, PitchOptions{}));
  REQUIRE_THROWS(createPitchExtractor(PitchMethod::Fcpe, "/nonexistent.onnx", SessionConfig{}, PitchOptions{}));
  REQUIRE_THROWS(pitchMethodFromString("crepe-huge"));
}
