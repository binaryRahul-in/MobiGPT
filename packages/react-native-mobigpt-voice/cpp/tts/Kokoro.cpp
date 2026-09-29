#include "Kokoro.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace mobigpt::tts {

KokoroVoice KokoroVoice::load(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open voice file " + path);
  const auto bytes = static_cast<size_t>(f.tellg());
  const size_t rowBytes = kKokoroStyleDim * sizeof(float);
  if (bytes == 0 || bytes % rowBytes != 0) {
    throw std::runtime_error("not a Kokoro voice (expected N x 256 float32): " + path);
  }
  std::vector<float> data(bytes / sizeof(float));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(bytes));
  if (!f) throw std::runtime_error("failed to read voice file " + path);
  return fromData(std::move(data));
}

KokoroVoice KokoroVoice::fromData(std::vector<float> data) {
  if (data.empty() || data.size() % kKokoroStyleDim != 0) throw std::runtime_error("voice data must be N x 256 floats");
  KokoroVoice v;
  v.data_ = std::move(data);
  return v;
}

const float* KokoroVoice::styleFor(size_t tokenCount) const {
  const size_t row = std::min(std::max<size_t>(tokenCount, 1), rows()) - 1;
  return data_.data() + row * kKokoroStyleDim;
}

namespace {

const rvc::TensorInfo* firstOfType(const std::vector<rvc::TensorInfo>& v, ONNXTensorElementDataType t) {
  for (const auto& i : v) {
    if (i.type == t) return &i;
  }
  return nullptr;
}

}  // namespace

KokoroModel::KokoroModel(const std::string& path, const rvc::SessionConfig& cfg) : model_(path, cfg) {
  for (const char* n : {"tokens", "input_ids"}) {
    if (model_.hasInput(n)) tokensName_ = n;
  }
  if (tokensName_.empty()) {
    const auto* t = firstOfType(model_.inputs(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    if (!t) throw std::runtime_error("not a Kokoro model: no int64 token input");
    tokensName_ = t->name;
  }
  if (!model_.hasInput("style")) throw std::runtime_error("not a Kokoro model: missing 'style' input");
  styleName_ = "style";
  if (!model_.hasInput("speed")) throw std::runtime_error("not a Kokoro model: missing 'speed' input");
  speedName_ = "speed";
  speedType_ = model_.input("speed")->type;
  for (const char* n : {"audio", "waveform"}) {
    if (model_.hasOutput(n)) outputName_ = n;
  }
  if (outputName_.empty()) outputName_ = model_.outputs().at(0).name;
}

std::vector<float> KokoroModel::infer(const std::vector<int64_t>& tokens, const float* style, float speed) {
  if (tokens.empty()) return {};
  if (tokens.size() > kKokoroMaxTokens) throw std::runtime_error("too many phoneme tokens in one window");
  std::vector<int64_t> ids;
  ids.reserve(tokens.size() + 2);
  ids.push_back(0);
  ids.insert(ids.end(), tokens.begin(), tokens.end());
  ids.push_back(0);
  std::vector<float> styleCopy(style, style + kKokoroStyleDim);

  std::vector<Ort::Value> in;
  in.push_back(rvc::borrowTensor(ids.data(), ids.size(), {1, static_cast<int64_t>(ids.size())}));
  in.push_back(rvc::borrowTensor(styleCopy.data(), styleCopy.size(), {1, static_cast<int64_t>(kKokoroStyleDim)}));
  float speedF = speed;
  int32_t speedI = static_cast<int32_t>(std::lround(speed));
  int64_t speedL = speedI;
  if (speedType_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    in.push_back(rvc::borrowTensor(&speedI, 1, {1}));
  } else if (speedType_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    in.push_back(rvc::borrowTensor(&speedL, 1, {1}));
  } else {
    in.push_back(rvc::borrowTensor(&speedF, 1, {1}));
  }
  const std::vector<const char*> inNames{tokensName_.c_str(), styleName_.c_str(), speedName_.c_str()};
  const std::vector<const char*> outNames{outputName_.c_str()};
  auto out = model_.run(inNames, in, outNames);
  return rvc::toFloatVector(out.at(0));
}

std::vector<float> synthesize(KokoroModel& model, const KokoroVoice& voice, const std::vector<std::vector<int64_t>>& windows,
                              const std::vector<float>& pausesAfter, float speed, SynthesisTimings* timings) {
  std::vector<float> audio;
  SynthesisTimings t;
  for (size_t i = 0; i < windows.size(); ++i) {
    const auto& w = windows[i];
    if (!w.empty()) {
      const auto t0 = std::chrono::steady_clock::now();
      auto chunk = model.infer(w, voice.styleFor(w.size()), speed);
      t.inferMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      ++t.windows;
      audio.insert(audio.end(), chunk.begin(), chunk.end());
    }
    const float pause = i < pausesAfter.size() ? pausesAfter[i] : 0.f;
    if (pause > 0) audio.resize(audio.size() + static_cast<size_t>(std::lround(pause * kKokoroSampleRate)), 0.f);
  }
  if (timings) *timings = t;
  return audio;
}

}  // namespace mobigpt::tts
