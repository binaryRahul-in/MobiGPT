#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

#include "Audio.h"
#include "AudioIO.h"
#include "Dsp.h"
#include "Pipeline.h"
#include "VoiceService.h"
#include "testing.h"

using namespace mobigpt::rvc;

namespace {

std::vector<float> voice(double hz, double seconds, int sr = 16000) {
  std::vector<float> x(static_cast<size_t>(seconds * sr));
  double ph = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    ph += 2 * kPi * hz / sr;
    double s = 0;
    for (int h = 1; h <= 6; ++h) s += std::sin(h * ph) / h;
    x[i] = static_cast<float>(0.25 * s);
  }
  return x;
}

double measuredPitch(const std::vector<float>& y, int sr) {
  auto x16 = resample(y.data(), y.size(), sr, 16000);
  auto ex = createPitchExtractor(PitchMethod::Dio, "", SessionConfig{}, PitchOptions{});
  auto f0 = ex->extract(x16.data(), x16.size(), x16.size() / 160);
  std::vector<float> v;
  for (float f : f0) if (f > 0) v.push_back(f);
  if (v.empty()) return 0;
  std::nth_element(v.begin(), v.begin() + static_cast<long>(v.size() / 2), v.end());
  return v[v.size() / 2];
}

double maxJump(const std::vector<float>& y) {
  double j = 0;
  for (size_t i = 1; i < y.size(); ++i) j = std::max(j, std::fabs(static_cast<double>(y[i]) - y[i - 1]));
  return j;
}

PipelineConfig baseConfig(const char* enc, const char* synth, PitchMethod pm = PitchMethod::Dio, const char* pitchModel = "") {
  PipelineConfig c;
  c.encoderPath = mt::fixture(enc);
  c.synthPath = mt::fixture(synth);
  c.pitchMethod = pm;
  c.pitchModelPath = *pitchModel ? mt::fixture(pitchModel) : "";
  c.session.accelerator = Accelerator::Cpu;
  c.rmsMixRate = 1.0f;  // keep the synthetic tone's amplitude for jump checks
  return c;
}

}  // namespace

TEST(stitcher_is_lossless_for_identity_windows) {
  // With an identity "model" the SOLA stitcher must reproduce the input bit
  // for bit, regardless of how input is split into blocks.
  PipelineConfig cfg;
  cfg.chunkSeconds = 0.5f;
  cfg.contextSeconds = 0.1f;
  cfg.crossfadeMs = 40.0f;
  cfg.solaSearchMs = 10.0f;
  auto plan = StreamingConverter::makePlan(cfg);
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> u(-1, 1);
  std::vector<float> x(16000 * 3 + 123);
  for (auto& v : x) v = u(rng);
  std::vector<float> out;
  StreamingConverter sc(plan, 160, [](const float* w, size_t n) { return std::vector<float>(w, w + n); },
                        [&](const float* y, size_t n) { out.insert(out.end(), y, y + n); });
  std::uniform_int_distribution<size_t> blk(1, 5000);
  for (size_t off = 0; off < x.size();) {
    size_t n = std::min(blk(rng), x.size() - off);
    sc.push(x.data() + off, n);
    off += n;
  }
  sc.flush();
  REQUIRE(out.size() == (x.size() + 159) / 160 * 160);
  for (size_t i = 0; i < x.size(); ++i) REQUIRE_NEAR(out[i], x[i], 1e-6);
}

TEST(all_model_layouts_load_and_convert) {
  SKIP_UNLESS_FIXTURE("synth_webui_40k.onnx");
  struct Combo { const char* enc; const char* synth; int sr; };
  for (auto c : {Combo{"enc_webui.onnx", "synth_webui_40k.onnx", 40000}, Combo{"enc_hf.onnx", "synth_webui_40k.onnx", 40000},
                 Combo{"enc_wokada.onnx", "synth_wokada_48k_fp16.onnx", 48000}, Combo{"enc_webui.onnx", "synth_nof0_32k.onnx", 32000},
                 Combo{"enc_v1.onnx", "synth_v1_40k.onnx", 40000}}) {
    RvcEngine e(baseConfig(c.enc, c.synth));
    e.load();
    REQUIRE(e.outputSampleRate() == c.sr);
    auto x = voice(200, 1.0);
    auto y = e.convertWindow(x.data(), x.size());
    REQUIRE(y.size() == x.size() / 160 * static_cast<size_t>(c.sr / 100));
    for (float v : y) REQUIRE(std::isfinite(v));
  }
}

TEST(pitch_shift_is_applied) {
  SKIP_UNLESS_FIXTURE("synth_webui_40k.onnx");
  for (float key : {0.0f, 12.0f, -5.0f}) {
    auto cfg = baseConfig("enc_webui.onnx", "synth_webui_40k.onnx", PitchMethod::Harvest);
    cfg.f0UpKey = key;
    RvcEngine e(cfg);
    e.load();
    auto x = voice(200, 2.0);
    std::vector<float> y;
    e.convertBuffer(x.data(), x.size(), [&](const float* s, size_t n) { y.insert(y.end(), s, s + n); });
    const double expect = 200.0 * std::pow(2.0, key / 12.0);
    const double got = measuredPitch(y, 40000);
    std::printf("    key %+.0f: expect %.1f Hz got %.1f Hz\n", key, expect, got);
    REQUIRE(std::fabs(got - expect) / expect < 0.03);
  }
}

TEST(chunked_output_is_continuous_and_exact_length) {
  SKIP_UNLESS_FIXTURE("synth_webui_40k.onnx");
  for (float chunk : {2.0f, 2.5f, 3.0f}) {
    auto cfg = baseConfig("enc_webui.onnx", "synth_webui_40k.onnx", PitchMethod::Pm);
    cfg.chunkSeconds = chunk;
    RvcEngine e(cfg);
    e.load();
    auto x = voice(180, 7.3);
    std::vector<float> y;
    auto st = e.convertBuffer(x.data(), x.size(), [&](const float* s, size_t n) { y.insert(y.end(), s, s + n); });
    REQUIRE(y.size() == (x.size() + 159) / 160 * 400);
    REQUIRE(st.chunks >= 3);
    const double jump = maxJump(y);
    std::printf("    chunk %.1fs: %d chunks, max sample jump %.4f, RTF %.3f\n", chunk, st.chunks, jump, st.realtimeFactor);
    // 0.3-amp sine at <=400 Hz @40k moves < 0.02/sample; a hard splice would jump up to 0.6.
    REQUIRE(jump < 0.08);
  }
}

TEST(sequential_strategy_matches_resident) {
  SKIP_UNLESS_FIXTURE("synth_webui_40k.onnx");
  auto x = voice(220, 4.0);
  std::vector<float> a, b;
  auto cfg = baseConfig("enc_webui.onnx", "synth_webui_40k.onnx", PitchMethod::Rmvpe, "rmvpe_mel.onnx");
  cfg.pitch.f0Min = 30;
  cfg.pitch.f0Max = 2000;
  RvcEngine r(cfg);
  r.load();
  r.convertBuffer(x.data(), x.size(), [&](const float* s, size_t n) { a.insert(a.end(), s, s + n); });
  cfg.strategy = LoadStrategy::Sequential;
  RvcEngine q(cfg);
  q.load();
  q.convertBuffer(x.data(), x.size(), [&](const float* s, size_t n) { b.insert(b.end(), s, s + n); });
  REQUIRE(a.size() == b.size());
  double maxDiff = 0;
  for (size_t i = 0; i < a.size(); ++i) maxDiff = std::max(maxDiff, std::fabs(static_cast<double>(a[i]) - b[i]));
  // Only difference: features stored as fp16 between stages.
  REQUIRE(maxDiff < 1e-2);
}

TEST(wokada_fp16_voice_with_waveform_rmvpe) {
  SKIP_UNLESS_FIXTURE("synth_wokada_48k_fp16.onnx");
  auto cfg = baseConfig("enc_wokada.onnx", "synth_wokada_48k_fp16.onnx", PitchMethod::Rmvpe, "rmvpe_wave.onnx");
  cfg.f0UpKey = 7;
  RvcEngine e(cfg);
  e.load();
  auto info = e.info();
  REQUIRE(info.synthLayout == "w-okada");
  REQUIRE(info.sampleRate == 48000);
  auto x = voice(150, 2.0);
  std::vector<float> y;
  e.convertBuffer(x.data(), x.size(), [&](const float* s, size_t n) { y.insert(y.end(), s, s + n); });
  REQUIRE_NEAR(measuredPitch(y, 48000), 220.0 * std::pow(2.0, 7 / 12.0), 220.0 * 0.03);
}

TEST(encoder_width_mismatch_is_reported) {
  SKIP_UNLESS_FIXTURE("synth_v1_40k.onnx");
  RvcEngine e(baseConfig("enc_webui.onnx", "synth_v1_40k.onnx"));
  e.load();
  auto x = voice(200, 0.5);
  bool threw = false;
  try {
    e.convertWindow(x.data(), x.size());
  } catch (const std::exception& ex) {
    threw = std::string(ex.what()).find("256") != std::string::npos;
  }
  REQUIRE(threw);
}

TEST(index_rate_is_forced_to_zero) {
  SKIP_UNLESS_FIXTURE("synth_webui_40k.onnx");
  auto cfg = baseConfig("enc_webui.onnx", "synth_webui_40k.onnx");
  cfg.indexRate = 0.75f;
  RvcEngine e(cfg);
  e.load();
  REQUIRE(e.config().indexRate == 0.0f);
  REQUIRE(!e.info().indexUsed);
  bool warned = false;
  for (auto& w : e.info().warnings) warned |= w.find("index_rate") != std::string::npos;
  REQUIRE(warned);
}

TEST(unavailable_accelerator_falls_back_to_cpu) {
  SKIP_UNLESS_FIXTURE("synth_webui_40k.onnx");
  for (auto acc : {Accelerator::Nnapi, Accelerator::CoreMl, Accelerator::Qnn, Accelerator::Xnnpack}) {
    SessionConfig sc;
    sc.accelerator = acc;
    OnnxModel m(mt::fixture("enc_webui.onnx"), sc);
    // Linux CI builds have none of these; they must degrade, not throw.
    REQUIRE(m.provider() == "cpu" || m.provider() == acceleratorName(acc));
  }
}

TEST(file_conversion_end_to_end) {
  SKIP_UNLESS_FIXTURE("synth_webui_40k.onnx");
  const std::string in = "/tmp/mobigpt_e2e_in.wav", out = "/tmp/mobigpt_e2e_out.wav";
  auto x = voice(200, 3.0, 44100);
  writeWavPcm16(in, x.data(), x.size(), 44100);
  auto& svc = VoiceService::instance();
  auto cfg = baseConfig("enc_webui.onnx", "synth_webui_40k.onnx", PitchMethod::Fcpe, "fcpe.onnx");
  svc.load(cfg);
  double lastProgress = -1;
  bool monotonic = true;
  auto st = svc.convertFile(in, out, [&](double p) {
    monotonic &= p >= lastProgress;
    lastProgress = p;
  });
  REQUIRE(monotonic && lastProgress == 1.0);
  auto y = readWav(out);
  REQUIRE(y.sampleRate == 40000);
  REQUIRE_NEAR(y.durationSeconds(), 3.0, 0.02);
  REQUIRE_NEAR(st.inputSeconds, 3.0, 0.01);
  std::remove(in.c_str());
  std::remove(out.c_str());
}

TEST(model_inspection_classifies_files) {
  SKIP_UNLESS_FIXTURE("synth_webui_40k.onnx");
  auto& svc = VoiceService::instance();
  REQUIRE(svc.inspect(mt::fixture("synth_webui_40k.onnx")).kind == "synthesizer");
  REQUIRE(svc.inspect(mt::fixture("synth_wokada_48k_fp16.onnx")).synth.sampleRate == 48000);
  REQUIRE(svc.inspect(mt::fixture("synth_v1_40k.onnx")).synth.channels == 256);
  REQUIRE(!svc.inspect(mt::fixture("synth_nof0_32k.onnx")).synth.usesF0);
  REQUIRE(svc.inspect(mt::fixture("enc_webui.onnx")).kind == "encoder");
  REQUIRE(svc.inspect(mt::fixture("enc_wokada.onnx")).kind == "encoder");
  REQUIRE(svc.inspect(mt::fixture("rmvpe_mel.onnx")).kind == "rmvpe");
  REQUIRE(svc.inspect(mt::fixture("rmvpe_wave.onnx")).kind == "rmvpe");
  REQUIRE(svc.inspect(mt::fixture("fcpe.onnx")).kind == "fcpe");
}

TEST(live_session_runs_natively_with_null_audio) {
  SKIP_UNLESS_FIXTURE("synth_webui_40k.onnx");
  std::atomic<size_t> played{0};
  mobigpt::audio::setHostOutputTap([&](const float*, size_t n, int sr) {
    if (sr == 40000) played += n;
  });
  auto& svc = VoiceService::instance();
  auto cfg = baseConfig("enc_webui.onnx", "synth_webui_40k.onnx", PitchMethod::Pm);
  cfg.chunkSeconds = 0.5f;
  cfg.contextSeconds = 0.1f;
  svc.load(cfg);
  svc.startLive();
  for (int i = 0; i < 60 && svc.liveStats().chunks < 2; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(50));
  auto st = svc.liveStats();
  svc.stopLive();
  mobigpt::audio::setHostOutputTap(nullptr);
  REQUIRE(st.error.empty());
  REQUIRE(st.chunks >= 2);
  REQUIRE(played.load() > 0);
  REQUIRE(!svc.liveStats().running);
}

TEST(cancel_stops_conversion) {
  SKIP_UNLESS_FIXTURE("synth_webui_40k.onnx");
  auto& svc = VoiceService::instance();
  svc.load(baseConfig("enc_webui.onnx", "synth_webui_40k.onnx", PitchMethod::Harvest));
  const std::string in = "/tmp/mobigpt_cancel_in.wav", out = "/tmp/mobigpt_cancel_out.wav";
  auto x = voice(200, 20.0);
  writeWavPcm16(in, x.data(), x.size(), 16000);
  auto st = svc.convertFile(in, out, [&](double p) {
    if (p > 0.1) svc.cancel();
  });
  REQUIRE(st.cancelled);
  REQUIRE(st.outputSeconds < 19.0);
  svc.unload();
  REQUIRE(!svc.isLoaded());
  std::remove(in.c_str());
  std::remove(out.c_str());
}
