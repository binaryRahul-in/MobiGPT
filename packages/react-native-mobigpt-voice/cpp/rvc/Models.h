// MobiGPT Voice — RVC neural components with I/O auto-detection.
//
// The community ships RVC ONNX files in (at least) three layouts; we accept
// all of them so users can drop in whatever they already have:
//
//   Content encoder (HuBERT / ContentVec)
//     RVC-WebUI   : source[1,1,N]            -> embed[1,F,768|256]
//     transformers: input_values[1,N] (+mask)-> hidden_states[1,F,768]
//     w-okada     : audio[1,N]               -> unit12 / units9 / unit12s
//
//   Synthesiser (net_g)
//     RVC-WebUI   : phone, phone_lengths, pitch, pitchf, ds, rnd -> audio
//     w-okada     : feats, p_len, pitch, pitchf, sid              -> audio
//     (pitch/pitchf absent for "no-f0" voices; fp16 I/O supported)
#pragma once

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "OrtSession.h"

namespace mobigpt::rvc {

struct SynthInfo {
  int sampleRate = 0;       // 0 = unknown until warm-up
  int channels = 768;       // content feature width (768 = v2, 256 = v1)
  bool usesF0 = true;
  int embOutputLayer = 12;  // from w-okada metadata when present
  bool useFinalProj = false;
  std::string layout;       // "rvc-webui" | "w-okada"
  std::string version;      // "v1" | "v2" | ""
};

// Parses the handful of fields we need out of RVC metadata (either flat
// custom metadata keys or w-okada's JSON blob under "metadata").
SynthInfo parseSynthMetadata(const std::map<std::string, std::string>& meta);

class ContentEncoder {
 public:
  ContentEncoder(const std::string& path, const SessionConfig& cfg);
  // Returns [frames][channels] (row-major) at 50 Hz for 16 kHz input.
  std::vector<float> encode(const float* audio16k, size_t count, int wantChannels, const SynthInfo& synth,
                            int* framesOut, int* channelsOut);
  const OnnxModel& model() const { return model_; }

 private:
  std::string pickOutput(int wantChannels, const SynthInfo& synth) const;
  OnnxModel model_;
  enum class Layout { Source3d, InputValues, Audio2d } layout_;
  std::string inputName_;
};

class Synthesizer {
 public:
  Synthesizer(const std::string& path, const SessionConfig& cfg, int defaultSampleRate, uint32_t seed);

  // feats: [frames][channels] at 100 Hz; pitch/pitchf: [frames].
  std::vector<float> synthesize(const std::vector<float>& feats, int frames, int channels,
                                const std::vector<int64_t>& pitch, const std::vector<float>& pitchf,
                                int speakerId);
  const SynthInfo& info() const { return info_; }
  const OnnxModel& model() const { return model_; }
  // Samples produced per 10 ms frame (sampleRate / 100).
  int hop() const { return info_.sampleRate / 100; }

 private:
  OnnxModel model_;
  SynthInfo info_;
  bool webui_ = true;
  bool featsHalf_ = false;
  bool pitchfHalf_ = false;
  int noiseChannels_ = 192;
  std::string outputName_;
  std::mt19937 rng_;
};

// RVC's coarse pitch quantisation (mel scale, 50-1100 Hz -> 1..255).
std::vector<int64_t> coarsePitch(const std::vector<float>& f0);

// Semitone shift applied to voiced frames.
void shiftPitch(std::vector<float>& f0, float semitones);

}  // namespace mobigpt::rvc
