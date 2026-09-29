// MobiGPT Voice — pitch (F0) extraction.
//
// Every extractor returns F0 in Hz at a 10 ms frame rate (hop 160 @ 16 kHz),
// which is exactly the rate RVC's synthesiser consumes after the 2x feature
// upsample. 0 Hz marks unvoiced frames.
//
//  * rmvpe   — neural, most robust (ONNX, ~90 MB int8 / ~345 MB fp32)
//  * fcpe    — neural, ~10x lighter than RMVPE (ONNX, ~40 MB)
//  * harvest — WORLD, high quality DSP, slow, zero model RAM
//  * dio     — WORLD + StoneMask refinement, fast DSP, zero model RAM
//  * pm      — normalised-autocorrelation tracker, fastest, zero model RAM
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "OrtSession.h"

namespace mobigpt::rvc {

enum class PitchMethod { Rmvpe, Fcpe, Harvest, Dio, Pm };

PitchMethod pitchMethodFromString(const std::string& s);
const char* pitchMethodName(PitchMethod m);
bool pitchMethodNeedsModel(PitchMethod m);

struct PitchOptions {
  double f0Min = 50.0;
  double f0Max = 1100.0;
  float threshold = -1.0f;  // voicing threshold; <0 = extractor default
};

class PitchExtractor {
 public:
  virtual ~PitchExtractor() = default;
  // audio16k: mono 16 kHz. Returns exactly `frames` values (10 ms hop).
  virtual std::vector<float> extract(const float* audio16k, size_t count, size_t frames) = 0;
  virtual const char* name() const = 0;
};

// modelPath is required for rmvpe / fcpe and ignored for DSP methods.
std::unique_ptr<PitchExtractor> createPitchExtractor(PitchMethod method, const std::string& modelPath,
                                                     const SessionConfig& session, const PitchOptions& opts);

// ---- decoding helpers (exposed for unit tests) -----------------------------

// RVC RMVPE decode: local weighted average of ±4 bins around the argmax of a
// 360-bin salience map (20 cents/bin), threshold 0.03 by default.
std::vector<float> decodeRmvpeSalience(const float* salience, size_t frames, float threshold = 0.03f);

// FCPE "local_argmax" decode over a 360-bin latent spanning 32.70-1975.5 Hz.
std::vector<float> decodeFcpeLatent(const float* latent, size_t frames, float threshold = 0.006f);

// Normalised autocorrelation tracker (Praat-style "pm" method, simplified).
std::vector<float> pmPitch(const float* audio16k, size_t count, size_t frames, double f0Min, double f0Max);

}  // namespace mobigpt::rvc
