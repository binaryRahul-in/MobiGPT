// mobigpt-rvc — host CLI around the exact C++ engine that ships in the app.
// Used by CI to run real community RVC models end-to-end on Linux/macOS.
//
//   mobigpt-rvc convert --encoder hubert.onnx --voice voice.onnx --pitch rmvpe \
//       --pitch-model rmvpe.onnx --key 0 --chunk 2.5 in.wav out.wav
//   mobigpt-rvc inspect model.onnx
//   mobigpt-rvc pitch --method dio in.wav
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "Checkpoint.h"
#include "Pipeline.h"
#include "VoiceService.h"

using namespace mobigpt::rvc;

namespace {

void usage() {
  std::fprintf(stderr,
               "usage:\n"
               "  mobigpt-rvc convert --encoder E --voice V [--pitch rmvpe|fcpe|harvest|dio|pm] [--pitch-model P]\n"
               "                      [--key SEMITONES] [--chunk SEC] [--sequential] [--accel auto|cpu|xnnpack]\n"
               "                      [--rms-mix R] [--speaker ID] IN.wav OUT.wav\n"
               "  mobigpt-rvc inspect MODEL.onnx\n"
               "  mobigpt-rvc pitch --method M [--pitch-model P] IN.wav\n"
               "  mobigpt-rvc tts --model KOKORO.onnx --voice VOICE.bin --tokens \"ids,…;ids,…\" [--pauses \"s;s\"]\n"
               "                  [--speed 1.0] OUT.wav\n"
               "  mobigpt-rvc rvc-info VOICE.zip|VOICE.pth\n"
               "  mobigpt-rvc rvc-import --template TEMPLATE.onnx VOICE.zip|VOICE.pth OUT_DIR\n"
               "  mobigpt-rvc info\n");
}

const char* typeName(ONNXTensorElementDataType t) {
  switch (t) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "float32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "float16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "int64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return "int32";
    default: return "other";
  }
}

std::string shapeStr(const std::vector<int64_t>& s) {
  std::string r = "[";
  for (size_t i = 0; i < s.size(); ++i) r += (i ? "," : "") + (s[i] < 0 ? std::string("?") : std::to_string(s[i]));
  return r + "]";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    usage();
    return 2;
  }
  const std::string cmd = argv[1];
  std::map<std::string, std::string> opt;
  std::vector<std::string> pos;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--", 0) == 0) {
      if (a == "--sequential") {
        opt[a] = "1";
      } else if (i + 1 < argc) {
        opt[a] = argv[++i];
      }
    } else {
      pos.push_back(a);
    }
  }
  try {
    auto& svc = VoiceService::instance();
    if (cmd == "info") {
      std::printf("onnxruntime %s\nproviders:", ortVersion().c_str());
      for (auto& p : availableProviders()) std::printf(" %s", p.c_str());
      std::printf("\n");
      return 0;
    }
    if (cmd == "inspect" && pos.size() == 1) {
      auto r = svc.inspect(pos[0]);
      std::printf("{\"kind\":\"%s\",\"sizeMB\":%.1f,\"quantized\":%s", r.kind.c_str(), r.fileMB, r.quantized ? "true" : "false");
      if (r.kind == "synthesizer") {
        std::printf(",\"sampleRate\":%d,\"channels\":%d,\"f0\":%s,\"layout\":\"%s\"", r.synth.sampleRate, r.synth.channels,
                    r.synth.usesF0 ? "true" : "false", r.synth.layout.c_str());
      }
      std::printf(",\"inputs\":[");
      for (size_t i = 0; i < r.inputs.size(); ++i)
        std::printf("%s{\"name\":\"%s\",\"type\":\"%s\",\"shape\":\"%s\"}", i ? "," : "", r.inputs[i].name.c_str(),
                    typeName(r.inputs[i].type), shapeStr(r.inputs[i].shape).c_str());
      std::printf("],\"outputs\":[");
      for (size_t i = 0; i < r.outputs.size(); ++i)
        std::printf("%s{\"name\":\"%s\",\"type\":\"%s\",\"shape\":\"%s\"}", i ? "," : "", r.outputs[i].name.c_str(),
                    typeName(r.outputs[i].type), shapeStr(r.outputs[i].shape).c_str());
      std::printf("]}\n");
      return 0;
    }
    if (cmd == "pitch" && pos.size() == 1) {
      auto f0 = svc.analyzePitch(pos[0], opt.count("--method") ? opt["--method"] : "dio", opt["--pitch-model"]);
      for (size_t i = 0; i < f0.size(); ++i) std::printf("%.2f\t%.2f\n", i * 0.01, f0[i]);
      return 0;
    }
    if ((cmd == "rvc-info" && pos.size() == 1) || (cmd == "rvc-import" && pos.size() == 2)) {
      RvcCheckpointInfo in;
      std::string model;
      if (cmd == "rvc-info") {
        in = inspectRvcCheckpoint(pos[0]);
      } else {
        auto r = importRvcCheckpoint(pos[0], opt["--template"], pos[1]);
        in = r.info;
        model = r.modelPath;
      }
      std::printf("{\"pth\":\"%s\",\"version\":\"%s\",\"sampleRate\":%d,\"f0\":%s,\"speakers\":%d,\"featureDim\":%d,"
                  "\"dtype\":\"%s\",\"tensors\":%zu,\"hasIndex\":%s,\"template\":\"%s\",\"model\":\"%s\"}\n",
                  in.pthName.c_str(), in.version.c_str(), in.sampleRate, in.f0 ? "true" : "false", in.speakers, in.featureDim,
                  in.dtype.c_str(), in.tensors, in.hasIndex ? "true" : "false", rvcTemplateName(in).c_str(), model.c_str());
      return 0;
    }
    if (cmd == "tts" && pos.size() == 1) {
      // --tokens: windows separated by ';', ids by ','. --pauses: seconds after each window.
      auto split = [](const std::string& s, char sep) {
        std::vector<std::string> out;
        size_t start = 0;
        while (start <= s.size()) {
          const size_t end = s.find(sep, start);
          out.push_back(s.substr(start, end == std::string::npos ? std::string::npos : end - start));
          if (end == std::string::npos) break;
          start = end + 1;
        }
        return out;
      };
      std::vector<std::vector<int64_t>> windows;
      for (const auto& w : split(opt["--tokens"], ';')) {
        std::vector<int64_t> ids;
        for (const auto& t : split(w, ',')) {
          if (!t.empty()) ids.push_back(std::strtoll(t.c_str(), nullptr, 10));
        }
        windows.push_back(std::move(ids));
      }
      std::vector<float> pauses;
      if (opt.count("--pauses")) {
        for (const auto& p : split(opt["--pauses"], ';')) pauses.push_back(p.empty() ? 0.f : std::strtof(p.c_str(), nullptr));
      }
      SessionConfig sc;
      if (opt.count("--accel")) sc.accelerator = acceleratorFromString(opt["--accel"]);
      const auto info = svc.ttsLoad(opt["--model"], sc);
      std::fprintf(stderr, "tts provider %s\n", info.provider.c_str());
      const float speed = opt.count("--speed") ? std::strtof(opt["--speed"].c_str(), nullptr) : 1.0f;
      const auto r = svc.ttsSynthesize(windows, pauses, opt["--voice"], speed, pos[0]);
      std::printf("{\"seconds\":%.3f,\"sampleRate\":%d,\"inferMs\":%.1f,\"rtf\":%.4f,\"windows\":%zu}\n", r.seconds, r.sampleRate,
                  r.inferMs, r.realtimeFactor, r.windows);
      return 0;
    }
    if (cmd == "convert" && pos.size() == 2) {
      PipelineConfig c;
      c.encoderPath = opt["--encoder"];
      c.synthPath = opt["--voice"];
      c.pitchMethod = pitchMethodFromString(opt.count("--pitch") ? opt["--pitch"] : "rmvpe");
      c.pitchModelPath = opt["--pitch-model"];
      if (opt.count("--key")) c.f0UpKey = std::strtof(opt["--key"].c_str(), nullptr);
      if (opt.count("--chunk")) c.chunkSeconds = std::strtof(opt["--chunk"].c_str(), nullptr);
      if (opt.count("--rms-mix")) c.rmsMixRate = std::strtof(opt["--rms-mix"].c_str(), nullptr);
      if (opt.count("--speaker")) c.speakerId = std::atoi(opt["--speaker"].c_str());
      if (opt.count("--accel")) c.session.accelerator = acceleratorFromString(opt["--accel"]);
      if (opt.count("--sequential")) c.strategy = LoadStrategy::Sequential;
      auto info = svc.load(c);
      std::fprintf(stderr, "voice: %s %s %d Hz, %d-d features, f0=%d\n", info.synthLayout.c_str(), info.version.c_str(),
                   info.sampleRate, info.channels, info.usesF0);
      for (auto& p : info.providers) std::fprintf(stderr, "provider %s\n", p.c_str());
      for (auto& w : info.warnings) std::fprintf(stderr, "warning: %s\n", w.c_str());
      auto st = svc.convertFile(pos[0], pos[1], nullptr);
      std::printf(
          "{\"inputSeconds\":%.3f,\"outputSeconds\":%.3f,\"sampleRate\":%d,\"chunks\":%d,\"wallMs\":%.1f,\"rtf\":%.4f,"
          "\"encoderMs\":%.1f,\"pitchMs\":%.1f,\"synthMs\":%.1f}\n",
          st.inputSeconds, st.outputSeconds, st.outputSampleRate, st.chunks, st.wallMs, st.realtimeFactor,
          st.stages.encoderMs, st.stages.pitchMs, st.stages.synthMs);
      return 0;
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  usage();
  return 2;
}
