// MobiGPT Voice — end-to-end RVC pipeline with chunked streaming.
//
//   16 kHz audio ─┬─> ContentEncoder (HuBERT/ContentVec) ─ 2x upsample ─┐
//                 └─> PitchExtractor (RMVPE/FCPE/Harvest/DIO/PM) ─ shift ─┴─> Synthesizer (net_g) ─> 40/48 kHz
//
// Design constraints (mobile):
//   * index_rate is hard-wired to 0: the FAISS retrieval index is never
//     loaded, saving 100-500 MB of RAM per voice.
//   * Audio is processed in fixed 2-3 s windows (plus context) so peak memory
//     is flat regardless of input length; windows are stitched with a SOLA
//     (synchronised overlap-add) crossfade.
//   * All tensors stay in C++; JS only ever sees file paths and statistics.
#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "Models.h"
#include "Pitch.h"

namespace mobigpt::rvc {

enum class LoadStrategy {
  Resident,    // all models in RAM at once (fastest, needed for live mode)
  Sequential,  // one model at a time per stage (lowest peak RAM, file mode only)
};

struct PipelineConfig {
  std::string encoderPath;     // HuBERT / ContentVec .onnx
  std::string synthPath;       // voice (net_g) .onnx
  std::string pitchModelPath;  // rmvpe.onnx / fcpe.onnx (DSP methods ignore it)
  PitchMethod pitchMethod = PitchMethod::Rmvpe;
  PitchOptions pitch;
  SessionConfig session;
  LoadStrategy strategy = LoadStrategy::Resident;

  float f0UpKey = 0.0f;     // semitones
  int speakerId = 0;
  float rmsMixRate = 0.25f;  // 1 = keep model loudness, 0 = follow input envelope
  // Retrieval index blend. Always forced to 0 on mobile (see header comment);
  // kept in the struct so callers can pass desktop presets unchanged.
  float indexRate = 0.0f;

  float chunkSeconds = 2.5f;    // core window length (2-3 s recommended)
  float contextSeconds = 0.3f;  // extra audio on each side of every window
  float crossfadeMs = 60.0f;
  float solaSearchMs = 10.0f;
  uint32_t seed = 42;
  int defaultSampleRate = 40000;
};

struct StageTimings {
  double encoderMs = 0, pitchMs = 0, synthMs = 0, totalMs = 0;
};

struct ConversionStats {
  double inputSeconds = 0;
  double outputSeconds = 0;
  double wallMs = 0;
  double realtimeFactor = 0;  // wall / audio duration (<1 = faster than realtime)
  int chunks = 0;
  int outputSampleRate = 0;
  StageTimings stages;
  bool cancelled = false;
};

struct EngineInfo {
  int sampleRate = 0;
  int channels = 0;
  bool usesF0 = true;
  std::string synthLayout;
  std::string version;
  std::string pitchMethod;
  std::vector<std::string> providers;  // "<component>:<provider>"
  std::vector<std::string> warnings;
  bool indexUsed = false;
};

using EmitFn = std::function<void(const float* samples, size_t count)>;
using ProgressFn = std::function<void(double fraction)>;

class RvcEngine {
 public:
  explicit RvcEngine(PipelineConfig cfg);
  ~RvcEngine();

  void load();
  void unload();
  bool loaded() const { return loaded_; }
  EngineInfo info() const;
  const PipelineConfig& config() const { return cfg_; }
  void setPitchShift(float semitones) { cfg_.f0UpKey = semitones; }

  int outputSampleRate() const;

  // Converts one self-contained 16 kHz window; returns exactly
  // (count / 160) * hop output samples.
  std::vector<float> convertWindow(const float* audio16k, size_t count, StageTimings* t = nullptr);

  // Chunked conversion of a whole buffer. `emit` receives output audio in
  // order as soon as each chunk is stitched.
  ConversionStats convertBuffer(const float* audio16k, size_t count, const EmitFn& emit,
                                const ProgressFn& progress = nullptr, const std::atomic<bool>* cancel = nullptr);

  // Reads any-rate WAV, writes 16-bit WAV at the voice's native rate.
  ConversionStats convertFile(const std::string& inWav, const std::string& outWav,
                              const ProgressFn& progress = nullptr, const std::atomic<bool>* cancel = nullptr);

 private:
  friend class StreamingConverter;
  void ensureEncoder();
  void ensurePitch();
  void ensureSynth();
  std::vector<float> encodeFeatures(const float* audio, size_t count, size_t frames, StageTimings* t);
  std::vector<float> extractF0(const float* audio, size_t count, size_t frames, StageTimings* t);
  std::vector<float> synthesize(const std::vector<float>& feats, std::vector<float> f0, size_t frames, StageTimings* t);
  ConversionStats convertSequential(const float* audio16k, size_t count, const EmitFn& emit,
                                    const ProgressFn& progress, const std::atomic<bool>* cancel);

  PipelineConfig cfg_;
  std::unique_ptr<ContentEncoder> encoder_;
  std::unique_ptr<PitchExtractor> pitch_;
  std::unique_ptr<Synthesizer> synth_;
  SynthInfo synthInfo_;
  bool loaded_ = false;
  std::vector<std::string> warnings_;
  std::vector<std::string> providers_;
};

// Frame-exact chunk scheduler + SOLA stitcher. Works for both offline
// (push everything, then flush) and live (push mic blocks as they arrive).
class StreamingConverter {
 public:
  struct Plan {
    int core = 0;     // 16 kHz samples per chunk (multiple of 320)
    int context = 0;  // samples of context each side (multiple of 320)
    int xfade = 0;    // crossfade, in 16 kHz samples (multiple of 160)
    int sola = 0;     // SOLA search radius, 16 kHz samples (multiple of 160)
  };
  static Plan makePlan(const PipelineConfig& cfg);

  // `window` converts a 16 kHz window into output-rate audio of exactly
  // (len / 160) * hop samples. Injected so the stitcher is testable alone.
  using WindowFn = std::function<std::vector<float>(const float*, size_t)>;

  StreamingConverter(Plan plan, int outHop, WindowFn window, EmitFn emit);

  void push(const float* audio16k, size_t count);
  void flush();
  int chunksProcessed() const { return chunks_; }
  size_t pendingInput() const { return buffer_.size(); }
  // Output latency introduced by chunking, in 16 kHz samples.
  int latencySamples() const { return plan_.core + plan_.xfade + plan_.context + plan_.sola; }

 private:
  void processChunk();
  Plan plan_;
  int hopOut_;
  WindowFn window_;
  EmitFn emit_;
  std::vector<float> buffer_;  // starts at (chunkStart - context)
  std::vector<float> tail_;    // held-back crossfade region from previous chunk
  bool started_ = false;
  int chunks_ = 0;
  size_t totalIn_ = 0;
  size_t totalOutFrames_ = 0;  // 10 ms frames emitted so far
};

// Applies RVC's rms_mix_rate volume-envelope transfer in place.
void mixRms(const float* ref16k, size_t refCount, float* out, size_t outCount, int outRate, float rate);

}  // namespace mobigpt::rvc
