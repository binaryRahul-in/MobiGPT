// MobiGPT Voice — process-wide service that owns the RVC engine plus the
// live / record / playback audio sessions. All methods are thread-safe and
// blocking; the JSI layer runs them on worker threads.
#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../tts/Kokoro.h"
#include "Pipeline.h"

namespace mobigpt::audio {
class AudioInput;
class AudioOutput;
}  // namespace mobigpt::audio

namespace mobigpt::rvc {

struct LiveStats {
  bool running = false;
  double latencyMs = 0;      // algorithmic (chunk + context + crossfade)
  double lastChunkMs = 0;    // compute time of the most recent chunk
  double realtimeFactor = 0; // compute / audio; must stay < 1 for glitch-free output
  int chunks = 0;
  int droppedBlocks = 0;     // input dropped because the engine fell behind
  float inputLevel = 0;
  float outputLevel = 0;
  std::string error;
};

struct RecordingResult {
  std::string path;
  double seconds = 0;
  int sampleRate = 0;
  float peak = 0;
};

struct ModelInspection {
  std::string kind;  // "synthesizer" | "encoder" | "rmvpe" | "fcpe" | "unknown"
  std::vector<TensorInfo> inputs;
  std::vector<TensorInfo> outputs;
  std::map<std::string, std::string> metadata;
  SynthInfo synth;   // valid when kind == synthesizer
  double fileMB = 0;
  bool quantized = false;  // contains int8 weights (DynamicQuantizeLinear / MatMulInteger)
};

struct BenchmarkResult {
  double audioSeconds = 0;
  StageTimings stages;
  double realtimeFactor = 0;
  int outputSampleRate = 0;
  std::vector<std::string> providers;
};

struct TtsInfo {
  std::string provider;
  std::vector<std::string> warnings;
};

struct TtsResult {
  std::string path;
  double seconds = 0;
  int sampleRate = 0;
  double inferMs = 0;
  double realtimeFactor = 0;  // compute / audio
  size_t windows = 0;
};

class VoiceService {
 public:
  static VoiceService& instance();

  EngineInfo load(const PipelineConfig& cfg);
  void unload();
  bool isLoaded() const;
  EngineInfo info() const;
  void setPitchShift(float semitones);

  ConversionStats convertFile(const std::string& inWav, const std::string& outWav, const ProgressFn& progress);
  void cancel();

  // Mic -> RVC -> speaker, fully native. Requires Resident strategy.
  void startLive();
  void stopLive();
  LiveStats liveStats() const;

  void startRecording(const std::string& wavPath);
  RecordingResult stopRecording();
  float recordingLevel() const { return recordLevel_.load(); }
  bool isRecording() const { return recording_.load(); }

  // Plays a WAV file through the native sink; `onDone` fires on completion.
  void play(const std::string& wavPath, std::function<void(bool completed, std::string error)> onDone);
  void stopPlayback();
  bool isPlaying() const { return playing_.load(); }

  ModelInspection inspect(const std::string& onnxPath) const;
  std::vector<float> analyzePitch(const std::string& wavPath, const std::string& method, const std::string& modelPath);
  BenchmarkResult benchmark(double audioSeconds);

  // Neural TTS (Kokoro). Independent of the RVC engine: both can be loaded.
  TtsInfo ttsLoad(const std::string& modelPath, const SessionConfig& cfg);
  void ttsUnload();
  bool ttsIsLoaded() const;
  // `windows` are phoneme-token windows (<= 510 ids each); `pausesAfter` seconds of silence after each.
  TtsResult ttsSynthesize(const std::vector<std::vector<int64_t>>& windows, const std::vector<float>& pausesAfter,
                          const std::string& voicePath, float speed, const std::string& outWav);

 private:
  VoiceService() = default;
  ~VoiceService();

  mutable std::mutex engineMutex_;
  std::unique_ptr<RvcEngine> engine_;

  mutable std::mutex ttsMutex_;
  std::unique_ptr<tts::KokoroModel> tts_;
  std::string ttsVoicePath_;
  std::unique_ptr<tts::KokoroVoice> ttsVoice_;
  std::atomic<bool> cancel_{false};

  // live
  mutable std::mutex liveMutex_;
  std::unique_ptr<audio::AudioInput> liveIn_;
  std::unique_ptr<audio::AudioOutput> liveOut_;
  std::thread liveThread_;
  std::atomic<bool> liveRunning_{false};
  LiveStats liveStats_;

  // recording
  std::mutex recMutex_;
  std::unique_ptr<audio::AudioInput> recIn_;
  std::atomic<bool> recording_{false};
  std::atomic<float> recordLevel_{0.0f};
  std::string recPath_;
  std::vector<float> recBuffer_;
  int recRate_ = 16000;

  // playback
  std::mutex playMutex_;
  std::thread playThread_;
  std::unique_ptr<audio::AudioOutput> playOut_;
  std::atomic<bool> playing_{false};
};

}  // namespace mobigpt::rvc
