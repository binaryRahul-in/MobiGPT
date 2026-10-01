#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "Checkpoint.h"
#include "OrtSession.h"
#include "VoiceService.h"
#include "testing.h"

using namespace mobigpt::rvc;

// Fixtures (tools/make_test_fixtures.py): voice_fixture.zip holds a real torch.save()-format
// "My Voice/My Voice.pth" (DEFLATE-compressed, next to a FAISS .index) with an FP32 dec.gain = 0.25
// and a 2-row FP16 speaker table; rvc_template_v2_40k.onnx expects 3 speaker rows and renders
// sine(pitchf) * mean(dec.gain).

namespace {
std::vector<uint8_t> slurp(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), {}};
}
uint16_t at16(const std::vector<uint8_t>& b, size_t off) { return static_cast<uint16_t>(b[off] | (b[off + 1] << 8)); }
std::string tempDir(const char* name) {
  const std::string d = std::string("/tmp/mobigpt_test_") + name;
  std::remove((d + "/model.onnx").c_str());
  std::remove((d + "/weights.bin").c_str());
  std::string cmd = "mkdir -p '" + d + "'";
  REQUIRE(std::system(cmd.c_str()) == 0);
  return d;
}
}  // namespace

TEST(rvc_archive_is_described_without_reading_weights) {
  SKIP_UNLESS_FIXTURE("voice_fixture.zip");
  auto in = inspectRvcCheckpoint(mt::fixture("voice_fixture.zip"));
  REQUIRE(in.pthName == "My Voice.pth");
  REQUIRE(in.version == "v2");
  REQUIRE(in.sampleRate == 40000);
  REQUIRE(in.f0);
  REQUIRE(in.speakers == 2);
  REQUIRE(in.featureDim == 768);
  REQUIRE(in.tensors == 3);
  REQUIRE(in.hasIndex);
  REQUIRE(in.info == "fixture");
  REQUIRE(rvcTemplateName(in) == "rvc_template_v2_40k.onnx");
  // A bare .pth works the same way.
  REQUIRE(inspectRvcCheckpoint(mt::fixture("voice_fixture.pth")).sampleRate == 40000);
}

TEST(rvc_import_writes_fp16_weights_at_template_offsets) {
  SKIP_UNLESS_FIXTURE("voice_fixture.zip");
  const std::string out = tempDir("import");
  auto r = importRvcCheckpoint(mt::fixture("voice_fixture.zip"), mt::fixture("rvc_template_v2_40k.onnx"), out);
  REQUIRE(r.modelPath == out + "/model.onnx");
  auto w = slurp(out + "/weights.bin");
  REQUIRE(w.size() == r.weightsBytes);
  REQUIRE(w.size() == 128);  // two tensors, each padded to 64 bytes
  for (int i = 0; i < 4; ++i) REQUIRE(at16(w, i * 2) == 0x3400);   // FP32 0.25 -> FP16
  for (int i = 0; i < 4; ++i) REQUIRE(at16(w, 64 + i * 2) == 0x3C00);  // speaker rows 0-1 = 1.0
  REQUIRE(at16(w, 64 + 8) == 0 && at16(w, 64 + 10) == 0);           // padded third row
}

TEST(imported_voice_loads_and_uses_the_voice_weights) {
  SKIP_UNLESS_FIXTURE("voice_fixture.zip");
  const std::string out = tempDir("run");
  auto r = importRvcCheckpoint(mt::fixture("voice_fixture.zip"), mt::fixture("rvc_template_v2_40k.onnx"), out);
  OnnxModel m(r.modelPath, SessionConfig{});
  const int64_t T = 50;
  std::vector<float> phone(static_cast<size_t>(T) * 768, 0.f), pitchf(static_cast<size_t>(T), 220.f), rnd(192 * static_cast<size_t>(T), 0.f);
  std::vector<int64_t> len{T}, pitch(static_cast<size_t>(T), 60), ds{0};
  std::vector<Ort::Value> in;
  in.push_back(borrowTensor(phone.data(), phone.size(), {1, T, 768}));
  in.push_back(borrowTensor(len.data(), 1, {1}));
  in.push_back(borrowTensor(pitch.data(), pitch.size(), {1, T}));
  in.push_back(borrowTensor(pitchf.data(), pitchf.size(), {1, T}));
  in.push_back(borrowTensor(ds.data(), 1, {1}));
  in.push_back(borrowTensor(rnd.data(), rnd.size(), {1, 192, T}));
  auto outs = m.run({"phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"}, in, {"audio"});
  auto audio = toFloatVector(outs[0]);
  REQUIRE(audio.size() == static_cast<size_t>(T) * 400);
  float peak = 0;
  for (float v : audio) peak = std::max(peak, std::fabs(v));
  REQUIRE_NEAR(peak, 0.25, 0.01);
  // The inspector sees an ordinary RVC-WebUI voice.
  auto ins = VoiceService::instance().inspect(r.modelPath);
  REQUIRE(ins.kind == "synthesizer");
  REQUIRE(ins.synth.sampleRate == 40000);
}

TEST(rvc_import_rejects_what_it_cannot_use) {
  SKIP_UNLESS_FIXTURE("voice_fixture.zip");
  const std::string out = tempDir("reject");
  REQUIRE_THROWS(importRvcCheckpoint(mt::fixture("voice_fixture.zip"), mt::fixture("kokoro.onnx"), out));  // not a template
  REQUIRE_THROWS(inspectRvcCheckpoint(mt::fixture("kokoro_voice.bin")));                                   // not a ZIP
  REQUIRE_THROWS(inspectRvcCheckpoint(mt::fixture("enc_webui.onnx")));                                     // ONNX, not a checkpoint
}
