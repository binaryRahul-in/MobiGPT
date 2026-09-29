#include "Models.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace mobigpt::rvc {

namespace {

// Minimal "find a scalar in a flat JSON object" helper. RVC metadata blobs
// are tiny and flat, so a full JSON parser would be dead weight.
bool jsonScalar(const std::string& json, const std::string& key, std::string& out) {
  const std::string needle = "\"" + key + "\"";
  size_t p = json.find(needle);
  if (p == std::string::npos) return false;
  p = json.find(':', p + needle.size());
  if (p == std::string::npos) return false;
  ++p;
  while (p < json.size() && (json[p] == ' ' || json[p] == '\t' || json[p] == '\n')) ++p;
  if (p >= json.size()) return false;
  if (json[p] == '"') {
    const size_t e = json.find('"', p + 1);
    if (e == std::string::npos) return false;
    out = json.substr(p + 1, e - p - 1);
    return true;
  }
  size_t e = p;
  while (e < json.size() && json[e] != ',' && json[e] != '}' && json[e] != '\n') ++e;
  out = json.substr(p, e - p);
  while (!out.empty() && (out.back() == ' ' || out.back() == '\r')) out.pop_back();
  return true;
}

bool truthy(const std::string& v) { return v == "true" || v == "1" || v == "True"; }

}  // namespace

SynthInfo parseSynthMetadata(const std::map<std::string, std::string>& meta) {
  SynthInfo info;
  auto flat = [&](const char* k) -> const std::string* {
    auto it = meta.find(k);
    return it == meta.end() ? nullptr : &it->second;
  };
  if (auto* v = flat("sample_rate")) info.sampleRate = std::atoi(v->c_str());
  if (auto* v = flat("sr")) info.sampleRate = std::atoi(v->c_str());
  if (auto* v = flat("f0")) info.usesF0 = truthy(*v);
  if (auto* v = flat("version")) info.version = *v;
  if (auto* v = flat("metadata")) {
    std::string s;
    if (jsonScalar(*v, "samplingRate", s)) info.sampleRate = std::atoi(s.c_str());
    if (jsonScalar(*v, "f0", s)) info.usesF0 = truthy(s);
    if (jsonScalar(*v, "embChannels", s)) info.channels = std::atoi(s.c_str());
    if (jsonScalar(*v, "embOutputLayer", s)) info.embOutputLayer = std::atoi(s.c_str());
    if (jsonScalar(*v, "useFinalProj", s)) info.useFinalProj = truthy(s);
    if (jsonScalar(*v, "version", s)) info.version = s;
  }
  // Snap odd values (e.g. "40k") to the rates RVC actually trains at.
  if (info.sampleRate > 0 && info.sampleRate < 1000) info.sampleRate *= 1000;
  return info;
}

// ------------------------------------------------------------------ encoder

ContentEncoder::ContentEncoder(const std::string& path, const SessionConfig& cfg) : model_(path, cfg) {
  const auto& in = model_.inputs().at(0);
  inputName_ = in.name;
  if (model_.hasInput("source") || in.shape.size() == 3) {
    layout_ = Layout::Source3d;
    if (model_.hasInput("source")) inputName_ = "source";
  } else if (model_.hasInput("input_values")) {
    layout_ = Layout::InputValues;
    inputName_ = "input_values";
  } else {
    layout_ = Layout::Audio2d;
  }
}

std::string ContentEncoder::pickOutput(int wantChannels, const SynthInfo& synth) const {
  const auto& outs = model_.outputs();
  if (model_.hasOutput("unit12") || model_.hasOutput("units9")) {
    // w-okada content_vec_500: choose the layer the voice was trained on.
    if (synth.embOutputLayer == 9 && model_.hasOutput("units9")) return "units9";
    if (synth.useFinalProj && model_.hasOutput("unit12s")) return "unit12s";
    if (wantChannels == 256 && model_.hasOutput("units9")) return "units9";
    if (model_.hasOutput("unit12")) return "unit12";
  }
  for (const auto& o : outs) {
    if (!o.shape.empty() && o.shape.back() == wantChannels) return o.name;
  }
  return outs.at(0).name;
}

std::vector<float> ContentEncoder::encode(const float* audio, size_t count, int wantChannels,
                                          const SynthInfo& synth, int* framesOut, int* channelsOut) {
  std::vector<float> wav(audio, audio + count);
  std::vector<int64_t> mask;
  std::vector<Ort::Value> ins;
  std::vector<const char*> names{inputName_.c_str()};
  const auto n = static_cast<int64_t>(count);
  if (layout_ == Layout::Source3d) {
    ins.push_back(borrowTensor(wav.data(), wav.size(), {1, 1, n}));
  } else {
    ins.push_back(borrowTensor(wav.data(), wav.size(), {1, n}));
    if (layout_ == Layout::InputValues && model_.hasInput("attention_mask")) {
      mask.assign(count, 1);
      ins.push_back(borrowTensor(mask.data(), mask.size(), {1, n}));
      names.push_back("attention_mask");
    }
  }
  const std::string out = pickOutput(wantChannels, synth);
  auto res = model_.run(names, ins, {out.c_str()});
  std::vector<int64_t> shape;
  std::vector<float> feats = toFloatVector(res[0], &shape);
  if (shape.size() != 3) throw std::runtime_error("content encoder: expected [1,frames,channels] output");
  int frames = static_cast<int>(shape[1]);
  int channels = static_cast<int>(shape[2]);
  if (channels != wantChannels && static_cast<int>(shape[1]) == wantChannels) {
    // [1, C, F] layout: transpose to [F][C].
    std::vector<float> t(feats.size());
    const int c = static_cast<int>(shape[1]), f = static_cast<int>(shape[2]);
    for (int i = 0; i < c; ++i)
      for (int j = 0; j < f; ++j) t[static_cast<size_t>(j) * c + i] = feats[static_cast<size_t>(i) * f + j];
    feats.swap(t);
    frames = f;
    channels = c;
  }
  if (framesOut) *framesOut = frames;
  if (channelsOut) *channelsOut = channels;
  return feats;
}

// -------------------------------------------------------------- synthesiser

Synthesizer::Synthesizer(const std::string& path, const SessionConfig& cfg, int defaultSampleRate, uint32_t seed)
    : model_(path, cfg), rng_(seed) {
  info_ = parseSynthMetadata(model_.metadata());
  webui_ = model_.hasInput("phone");
  if (!webui_ && !model_.hasInput("feats")) {
    throw std::runtime_error("not an RVC synthesiser: expected a 'phone' or 'feats' input");
  }
  info_.layout = webui_ ? "rvc-webui" : "w-okada";
  info_.usesF0 = model_.hasInput("pitchf");
  const TensorInfo* feats = model_.input(webui_ ? "phone" : "feats");
  if (feats->shape.size() == 3 && feats->shape[2] > 0) info_.channels = static_cast<int>(feats->shape[2]);
  featsHalf_ = feats->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  if (const TensorInfo* pf = model_.input("pitchf")) pitchfHalf_ = pf->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  if (const TensorInfo* rnd = model_.input("rnd")) {
    if (rnd->shape.size() == 3 && rnd->shape[1] > 0) noiseChannels_ = static_cast<int>(rnd->shape[1]);
  }
  outputName_ = model_.hasOutput("audio") ? "audio" : model_.outputs().at(0).name;
  if (info_.version.empty()) info_.version = info_.channels == 256 ? "v1" : "v2";

  // Warm-up with 20 frames: primes ORT kernels and lets us measure the true
  // hop (and therefore sample rate) even when metadata is missing.
  const int warmFrames = 20;
  std::vector<float> f(static_cast<size_t>(warmFrames) * info_.channels, 0.0f);
  std::vector<float> pf(static_cast<size_t>(warmFrames), 0.0f);
  std::vector<int64_t> p(static_cast<size_t>(warmFrames), 1);
  const int savedSr = info_.sampleRate;
  info_.sampleRate = 100;  // so hop() does not divide by zero during warm-up
  std::vector<float> y = synthesize(f, warmFrames, info_.channels, p, pf, 0);
  const int measuredHop = static_cast<int>(y.size() / warmFrames);
  info_.sampleRate = savedSr;
  if (measuredHop > 0 && measuredHop * warmFrames == static_cast<int>(y.size())) {
    const int measuredSr = measuredHop * 100;
    if (info_.sampleRate != measuredSr) info_.sampleRate = measuredSr;
  }
  if (info_.sampleRate <= 0) info_.sampleRate = defaultSampleRate;
}

std::vector<float> Synthesizer::synthesize(const std::vector<float>& feats, int frames, int channels,
                                           const std::vector<int64_t>& pitch, const std::vector<float>& pitchf,
                                           int speakerId) {
  if (frames <= 0) return {};
  if (static_cast<int>(feats.size()) != frames * channels) throw std::invalid_argument("feats size mismatch");
  if (channels != info_.channels) {
    throw std::runtime_error("content encoder width (" + std::to_string(channels) + ") does not match voice model (" +
                             std::to_string(info_.channels) + "); use a " + (info_.channels == 256 ? "v1 (256-d)" : "v2 (768-d)") +
                             " encoder");
  }
  const int64_t T = frames;
  std::vector<Ort::Value> ins;
  std::vector<const char*> names;

  std::vector<float> featsF;
  std::vector<uint16_t> featsH;
  if (featsHalf_) {
    featsH.resize(feats.size());
    for (size_t i = 0; i < feats.size(); ++i) featsH[i] = floatToHalf(feats[i]);
    ins.push_back(Ort::Value::CreateTensor(OnnxModel::cpuMemory(), featsH.data(), featsH.size() * 2,
                                           std::vector<int64_t>{1, T, channels}.data(), 3,
                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
  } else {
    featsF = feats;
    ins.push_back(borrowTensor(featsF.data(), featsF.size(), {1, T, channels}));
  }
  names.push_back(webui_ ? "phone" : "feats");

  int64_t len = T;
  ins.push_back(borrowTensor(&len, 1, {1}));
  names.push_back(webui_ ? "phone_lengths" : "p_len");

  std::vector<int64_t> pitchCopy;
  std::vector<float> pitchfCopy;
  std::vector<uint16_t> pitchfH;
  if (info_.usesF0) {
    pitchCopy.assign(pitch.begin(), pitch.end());
    pitchCopy.resize(static_cast<size_t>(T), 1);
    ins.push_back(borrowTensor(pitchCopy.data(), pitchCopy.size(), {1, T}));
    names.push_back("pitch");
    pitchfCopy.assign(pitchf.begin(), pitchf.end());
    pitchfCopy.resize(static_cast<size_t>(T), 0.0f);
    if (pitchfHalf_) {
      pitchfH.resize(pitchfCopy.size());
      for (size_t i = 0; i < pitchfCopy.size(); ++i) pitchfH[i] = floatToHalf(pitchfCopy[i]);
      ins.push_back(Ort::Value::CreateTensor(OnnxModel::cpuMemory(), pitchfH.data(), pitchfH.size() * 2,
                                             std::vector<int64_t>{1, T}.data(), 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
    } else {
      ins.push_back(borrowTensor(pitchfCopy.data(), pitchfCopy.size(), {1, T}));
    }
    names.push_back("pitchf");
  }

  int64_t sid = speakerId;
  ins.push_back(borrowTensor(&sid, 1, {1}));
  names.push_back(webui_ ? "ds" : "sid");

  std::vector<float> noise;
  std::vector<uint16_t> noiseH;
  if (const TensorInfo* rnd = model_.input("rnd")) {
    noise.resize(static_cast<size_t>(noiseChannels_) * T);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : noise) v = nd(rng_);
    if (rnd->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      noiseH.resize(noise.size());
      for (size_t i = 0; i < noise.size(); ++i) noiseH[i] = floatToHalf(noise[i]);
      ins.push_back(Ort::Value::CreateTensor(OnnxModel::cpuMemory(), noiseH.data(), noiseH.size() * 2,
                                             std::vector<int64_t>{1, noiseChannels_, T}.data(), 3,
                                             ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
    } else {
      ins.push_back(borrowTensor(noise.data(), noise.size(), {1, noiseChannels_, T}));
    }
    names.push_back("rnd");
  }

  auto outs = model_.run(names, ins, {outputName_.c_str()});
  std::vector<float> audio = toFloatVector(outs[0]);
  for (auto& v : audio) {
    if (!std::isfinite(v)) v = 0.0f;
  }
  return audio;
}

std::vector<int64_t> coarsePitch(const std::vector<float>& f0) {
  const double melMin = 1127.0 * std::log(1.0 + 50.0 / 700.0);
  const double melMax = 1127.0 * std::log(1.0 + 1100.0 / 700.0);
  std::vector<int64_t> out(f0.size(), 1);
  for (size_t i = 0; i < f0.size(); ++i) {
    double mel = 1127.0 * std::log(1.0 + f0[i] / 700.0);
    if (mel > 0.0) mel = (mel - melMin) * 254.0 / (melMax - melMin) + 1.0;
    mel = std::max(1.0, std::min(255.0, mel));
    out[i] = static_cast<int64_t>(std::lrint(mel));
  }
  return out;
}

void shiftPitch(std::vector<float>& f0, float semitones) {
  if (semitones == 0.0f) return;
  const float k = std::pow(2.0f, semitones / 12.0f);
  for (auto& v : f0) v *= k;
}

}  // namespace mobigpt::rvc
