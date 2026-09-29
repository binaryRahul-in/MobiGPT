// MobiGPT Voice — on-device neural text-to-speech with Kokoro-82M (Apache-2.0)
// on ONNX Runtime.
//
// Text → phonemes happens in TypeScript (misaki lexicons); this layer receives
// phoneme token ids and returns 24 kHz audio, so no tensors cross into JS.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "../rvc/OrtSession.h"

namespace mobigpt::tts {

constexpr int kKokoroSampleRate = 24000;
constexpr size_t kKokoroMaxTokens = 510;  // per window, excluding the two pad tokens
constexpr size_t kKokoroStyleDim = 256;

// A Kokoro voice: one 256-d style vector per phoneme count (row n-1 for n tokens).
// Stored as raw little-endian float32, `rows x 256` (the voices/*.bin files of
// onnx-community/Kokoro-82M-v1.0-ONNX, 510 rows = 522 KB).
class KokoroVoice {
 public:
  static KokoroVoice load(const std::string& path);
  static KokoroVoice fromData(std::vector<float> data);

  size_t rows() const { return data_.size() / kKokoroStyleDim; }
  const float* styleFor(size_t tokenCount) const;

 private:
  std::vector<float> data_;
};

// Accepts both published export layouts:
//   kokoro-onnx (thewh1teagle): tokens[1,T] int64, style[1,256], speed[1] float -> audio[N]
//   onnx-community:             input_ids[1,T] int64, style[1,256], speed[1] -> waveform[1,N]
class KokoroModel {
 public:
  KokoroModel(const std::string& path, const rvc::SessionConfig& cfg);

  // Synthesises one window. `tokens` excludes the pad ids; they are added here.
  std::vector<float> infer(const std::vector<int64_t>& tokens, const float* style, float speed);

  const std::string& provider() const { return model_.provider(); }
  const std::vector<std::string>& warnings() const { return model_.warnings(); }

 private:
  rvc::OnnxModel model_;
  std::string tokensName_, styleName_, speedName_, outputName_;
  ONNXTensorElementDataType speedType_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
};

struct SynthesisTimings {
  double inferMs = 0;
  size_t windows = 0;
};

// Synthesises consecutive windows and joins them with the given pauses
// (seconds of silence after each window; missing entries mean no pause).
std::vector<float> synthesize(KokoroModel& model, const KokoroVoice& voice, const std::vector<std::vector<int64_t>>& windows,
                              const std::vector<float>& pausesAfter, float speed, SynthesisTimings* timings = nullptr);

}  // namespace mobigpt::tts
