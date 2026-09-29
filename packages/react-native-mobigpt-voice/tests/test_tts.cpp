#include <cmath>
#include <cstdio>
#include <vector>

#include "../cpp/tts/Kokoro.h"
#include "Audio.h"
#include "VoiceService.h"
#include "testing.h"

using namespace mobigpt;

// Fixture model: audio = mean(style) / speed, 100 samples per input token (pads included).
// Fixture voice: row r is filled with r + 1, so k tokens select a style with mean k.

TEST(kokoro_voice_selects_style_row_by_token_count) {
  SKIP_UNLESS_FIXTURE("kokoro_voice.bin");
  auto v = tts::KokoroVoice::load(mt::fixture("kokoro_voice.bin"));
  REQUIRE(v.rows() == 510);
  REQUIRE_NEAR(v.styleFor(1)[0], 1.0, 0);
  REQUIRE_NEAR(v.styleFor(37)[255], 37.0, 0);
  REQUIRE_NEAR(v.styleFor(5000)[0], 510.0, 0);  // clamps to the last row
  REQUIRE_NEAR(v.styleFor(0)[0], 1.0, 0);
  REQUIRE_THROWS(tts::KokoroVoice::fromData(std::vector<float>(100)));
}

TEST(kokoro_both_export_layouts_pad_tokens_and_apply_speed) {
  SKIP_UNLESS_FIXTURE("kokoro.onnx");
  auto voice = tts::KokoroVoice::load(mt::fixture("kokoro_voice.bin"));
  for (const char* name : {"kokoro.onnx", "kokoro_community.onnx"}) {
    tts::KokoroModel m(mt::fixture(name), {});
    const std::vector<int64_t> tokens(12, 50);
    auto a = m.infer(tokens, voice.styleFor(tokens.size()), 1.0f);
    REQUIRE(a.size() == (tokens.size() + 2) * 100);  // two pad tokens added
    REQUIRE_NEAR(a[0], 12.0, 1e-5);
    auto fast = m.infer(tokens, voice.styleFor(tokens.size()), 2.0f);
    REQUIRE_NEAR(fast[0], 6.0, 1e-5);
  }
}

TEST(kokoro_synthesize_joins_windows_with_pauses) {
  SKIP_UNLESS_FIXTURE("kokoro.onnx");
  tts::KokoroModel m(mt::fixture("kokoro.onnx"), {});
  auto voice = tts::KokoroVoice::load(mt::fixture("kokoro_voice.bin"));
  std::vector<std::vector<int64_t>> windows{std::vector<int64_t>(3, 10), {}, std::vector<int64_t>(7, 10)};
  tts::SynthesisTimings t;
  auto audio = tts::synthesize(m, voice, windows, {0.5f, 0.f, 0.f}, 1.0f, &t);
  const size_t first = 5 * 100, pause = 12000, second = 9 * 100;
  REQUIRE(audio.size() == first + pause + second);
  REQUIRE(t.windows == 2);  // the empty window is skipped
  REQUIRE_NEAR(audio[0], 3.0, 1e-5);
  REQUIRE_NEAR(audio[first + pause / 2], 0.0, 0);
  REQUIRE_NEAR(audio.back(), 7.0, 1e-5);
}

TEST(tts_service_writes_24k_wav_and_rejects_bad_input) {
  SKIP_UNLESS_FIXTURE("kokoro.onnx");
  auto& svc = rvc::VoiceService::instance();
  REQUIRE_THROWS(svc.ttsSynthesize({{1, 2}}, {}, mt::fixture("kokoro_voice.bin"), 1.f, "/tmp/never.wav"));
  auto info = svc.ttsLoad(mt::fixture("kokoro.onnx"), {});
  REQUIRE(!info.provider.empty());
  REQUIRE(svc.ttsIsLoaded());
  const std::string out = "/tmp/mobigpt_tts_test.wav";
  auto r = svc.ttsSynthesize({std::vector<int64_t>(20, 5)}, {}, mt::fixture("kokoro_voice.bin"), 1.f, out);
  REQUIRE(r.sampleRate == 24000);
  REQUIRE_NEAR(r.seconds, 22 * 100 / 24000.0, 1e-6);
  auto wav = rvc::readWav(out);
  REQUIRE(wav.sampleRate == 24000);
  REQUIRE(wav.samples.size() == 2200);
  REQUIRE_THROWS(svc.ttsSynthesize({std::vector<int64_t>(600, 5)}, {}, mt::fixture("kokoro_voice.bin"), 1.f, out));
  REQUIRE_THROWS(svc.ttsSynthesize({{}}, {}, mt::fixture("kokoro_voice.bin"), 1.f, out));
  svc.ttsUnload();
  REQUIRE(!svc.ttsIsLoaded());
  std::remove(out.c_str());
}
