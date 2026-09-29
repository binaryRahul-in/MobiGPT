#include "VoiceService.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <fstream>
#include <stdexcept>

#include "../platform/AudioIO.h"
#include "Audio.h"
#include "Dsp.h"

namespace mobigpt::rvc {

namespace {
using Clock = std::chrono::steady_clock;

double fileSizeMB(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  return f ? static_cast<double>(f.tellg()) / (1024.0 * 1024.0) : 0.0;
}

bool fileContains(const std::string& path, const std::string& needle, size_t maxBytes) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::string buf(1 << 16, '\0');
  std::string carry;
  size_t read = 0;
  while (f && read < maxBytes) {
    f.read(&buf[0], static_cast<std::streamsize>(buf.size()));
    const auto n = static_cast<size_t>(f.gcount());
    if (!n) break;
    read += n;
    std::string hay = carry + buf.substr(0, n);
    if (hay.find(needle) != std::string::npos) return true;
    carry = hay.substr(hay.size() > needle.size() ? hay.size() - needle.size() : 0);
  }
  return false;
}
}  // namespace

VoiceService& VoiceService::instance() {
  static VoiceService* s = new VoiceService();  // intentionally leaked: outlives JS runtime teardown
  return *s;
}

VoiceService::~VoiceService() {
  stopLive();
  stopPlayback();
}

EngineInfo VoiceService::load(const PipelineConfig& cfg) {
  stopLive();
  std::lock_guard<std::mutex> lk(engineMutex_);
  engine_.reset();
  auto e = std::make_unique<RvcEngine>(cfg);
  e->load();
  engine_ = std::move(e);
  return engine_->info();
}

void VoiceService::unload() {
  stopLive();
  std::lock_guard<std::mutex> lk(engineMutex_);
  engine_.reset();
}

bool VoiceService::isLoaded() const {
  std::lock_guard<std::mutex> lk(engineMutex_);
  return engine_ && engine_->loaded();
}

EngineInfo VoiceService::info() const {
  std::lock_guard<std::mutex> lk(engineMutex_);
  if (!engine_) throw std::runtime_error("no voice loaded");
  return engine_->info();
}

void VoiceService::setPitchShift(float semitones) {
  std::lock_guard<std::mutex> lk(engineMutex_);
  if (engine_) engine_->setPitchShift(semitones);
}

ConversionStats VoiceService::convertFile(const std::string& inWav, const std::string& outWav, const ProgressFn& progress) {
  std::lock_guard<std::mutex> lk(engineMutex_);
  if (!engine_) throw std::runtime_error("no voice loaded");
  cancel_ = false;
  return engine_->convertFile(inWav, outWav, progress, &cancel_);
}

void VoiceService::cancel() { cancel_ = true; }

// --------------------------------------------------------------------- live

void VoiceService::startLive() {
  std::lock_guard<std::mutex> lk(liveMutex_);
  if (liveRunning_) return;
  {
    std::lock_guard<std::mutex> ek(engineMutex_);
    if (!engine_) throw std::runtime_error("load a voice before starting live mode");
    if (engine_->config().strategy != LoadStrategy::Resident) {
      throw std::runtime_error("live mode needs the 'resident' load strategy");
    }
  }
  liveStats_ = LiveStats{};
  liveIn_ = audio::createAudioInput();
  liveOut_ = audio::createAudioOutput();

  struct Shared {
    std::mutex m;
    std::condition_variable cv;
    std::deque<float> q;
    size_t cap = 16000 * 4;
    int dropped = 0;
    float level = 0;
  };
  auto shared = std::make_shared<Shared>();
  liveRunning_ = true;
  if (!liveIn_->start(16000, [shared](const float* s, size_t n) {
        std::lock_guard<std::mutex> l(shared->m);
        if (shared->q.size() + n > shared->cap) {
          const size_t drop = std::min(shared->q.size(), shared->q.size() + n - shared->cap);
          shared->q.erase(shared->q.begin(), shared->q.begin() + static_cast<long>(drop));
          ++shared->dropped;
        }
        shared->q.insert(shared->q.end(), s, s + n);
        shared->level = peak(s, n);
        shared->cv.notify_one();
      })) {
    liveRunning_ = false;
    const std::string err = liveIn_->error();
    liveIn_.reset();
    liveOut_.reset();
    throw std::runtime_error("microphone unavailable: " + err);
  }
  const int inRate = liveIn_->sampleRate();

  liveThread_ = std::thread([this, shared, inRate] {
    int outRate;
    PipelineConfig cfg;
    {
      std::lock_guard<std::mutex> ek(engineMutex_);
      outRate = engine_->outputSampleRate();
      cfg = engine_->config();
    }
    if (!liveOut_->start(outRate)) {
      std::lock_guard<std::mutex> l(liveMutex_);
      liveStats_.error = "speaker unavailable: " + liveOut_->error();
      liveRunning_ = false;
      return;
    }
    StreamResampler rs(inRate, 16000);
    auto plan = StreamingConverter::makePlan(cfg);
    {
      std::lock_guard<std::mutex> l(liveMutex_);
      liveStats_.running = true;
      liveStats_.latencyMs = (plan.core + plan.xfade + plan.context + plan.sola) / 16.0;
    }
    StreamingConverter sc(
        plan, outRate / 100,
        [this, &plan](const float* w, size_t n) {
          const auto t0 = Clock::now();
          std::vector<float> y;
          {
            std::lock_guard<std::mutex> ek(engineMutex_);
            if (!engine_) return std::vector<float>((n / 160) * 1, 0.0f);
            y = engine_->convertWindow(w, n, nullptr);
          }
          const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
          std::lock_guard<std::mutex> l(liveMutex_);
          liveStats_.lastChunkMs = ms;
          liveStats_.realtimeFactor = ms / (plan.core / 16.0);
          liveStats_.chunks++;
          return y;
        },
        [this](const float* y, size_t n) {
          {
            std::lock_guard<std::mutex> l(liveMutex_);
            liveStats_.outputLevel = peak(y, n);
          }
          liveOut_->write(y, n);
        });
    std::vector<float> block, x16;
    try {
      while (liveRunning_) {
        {
          std::unique_lock<std::mutex> l(shared->m);
          shared->cv.wait_for(l, std::chrono::milliseconds(50), [&] { return !shared->q.empty() || !liveRunning_; });
          block.assign(shared->q.begin(), shared->q.end());
          shared->q.clear();
          std::lock_guard<std::mutex> sl(liveMutex_);
          liveStats_.droppedBlocks = shared->dropped;
          liveStats_.inputLevel = shared->level;
        }
        if (block.empty()) continue;
        x16.clear();
        rs.process(block.data(), block.size(), x16);
        sc.push(x16.data(), x16.size());
      }
    } catch (const std::exception& e) {
      std::lock_guard<std::mutex> l(liveMutex_);
      liveStats_.error = e.what();
    }
    liveOut_->stop();
    std::lock_guard<std::mutex> l(liveMutex_);
    liveStats_.running = false;
  });
}

void VoiceService::stopLive() {
  std::unique_ptr<audio::AudioInput> in;
  {
    std::lock_guard<std::mutex> lk(liveMutex_);
    if (!liveRunning_ && !liveThread_.joinable()) return;
    liveRunning_ = false;
    in = std::move(liveIn_);
  }
  if (in) in->stop();
  if (liveThread_.joinable()) liveThread_.join();
  std::lock_guard<std::mutex> lk(liveMutex_);
  liveOut_.reset();
  liveStats_.running = false;
}

LiveStats VoiceService::liveStats() const {
  std::lock_guard<std::mutex> lk(liveMutex_);
  return liveStats_;
}

// ---------------------------------------------------------------- recording

void VoiceService::startRecording(const std::string& wavPath) {
  std::lock_guard<std::mutex> lk(recMutex_);
  if (recording_) throw std::runtime_error("already recording");
  recIn_ = audio::createAudioInput();
  recBuffer_.clear();
  recPath_ = wavPath;
  recording_ = true;
  if (!recIn_->start(16000, [this](const float* s, size_t n) {
        // Cap at 10 minutes to protect memory on low-end devices.
        if (recBuffer_.size() < static_cast<size_t>(recRate_) * 600) recBuffer_.insert(recBuffer_.end(), s, s + n);
        recordLevel_ = peak(s, n);
      })) {
    recording_ = false;
    const std::string err = recIn_->error();
    recIn_.reset();
    throw std::runtime_error("microphone unavailable: " + err);
  }
  recRate_ = recIn_->sampleRate();
}

RecordingResult VoiceService::stopRecording() {
  std::lock_guard<std::mutex> lk(recMutex_);
  if (!recording_) throw std::runtime_error("not recording");
  recIn_->stop();
  recIn_.reset();
  recording_ = false;
  recordLevel_ = 0;
  RecordingResult r;
  r.path = recPath_;
  r.sampleRate = recRate_;
  r.seconds = recRate_ > 0 ? static_cast<double>(recBuffer_.size()) / recRate_ : 0;
  r.peak = peak(recBuffer_.data(), recBuffer_.size());
  writeWavPcm16(recPath_, recBuffer_.data(), recBuffer_.size(), recRate_);
  recBuffer_.clear();
  recBuffer_.shrink_to_fit();
  return r;
}

// ----------------------------------------------------------------- playback

void VoiceService::play(const std::string& wavPath, std::function<void(bool, std::string)> onDone) {
  stopPlayback();
  AudioBuffer buf = readWav(wavPath);
  std::lock_guard<std::mutex> lk(playMutex_);
  playOut_ = audio::createAudioOutput();
  if (!playOut_->start(buf.sampleRate)) {
    const std::string err = playOut_->error();
    playOut_.reset();
    throw std::runtime_error("speaker unavailable: " + err);
  }
  playing_ = true;
  playThread_ = std::thread([this, b = std::move(buf), onDone = std::move(onDone)] {
    const size_t block = static_cast<size_t>(b.sampleRate / 20);  // 50 ms
    bool completed = true;
    for (size_t off = 0; off < b.samples.size(); off += block) {
      if (!playing_ || !playOut_->write(b.samples.data() + off, std::min(block, b.samples.size() - off))) {
        completed = false;
        break;
      }
    }
    if (completed) playOut_->drain();
    playing_ = false;
    if (onDone) onDone(completed, completed ? "" : "stopped");
  });
}

void VoiceService::stopPlayback() {
  std::lock_guard<std::mutex> lk(playMutex_);
  playing_ = false;
  if (playOut_) playOut_->stop();
  if (playThread_.joinable()) playThread_.join();
  playOut_.reset();
}

// -------------------------------------------------------------------- tools

ModelInspection VoiceService::inspect(const std::string& path) const {
  SessionConfig sc;
  sc.accelerator = Accelerator::Cpu;
  sc.lowMemory = true;
  OnnxModel m(path, sc);
  ModelInspection r;
  r.inputs = m.inputs();
  r.outputs = m.outputs();
  r.metadata = m.metadata();
  r.fileMB = fileSizeMB(path);
  r.quantized = fileContains(path, "MatMulInteger", 64u << 20) || fileContains(path, "DynamicQuantizeLinear", 64u << 20) ||
                fileContains(path, "QLinearConv", 64u << 20);
  if (m.hasInput("phone") || m.hasInput("feats")) {
    r.kind = "synthesizer";
    r.synth = parseSynthMetadata(m.metadata());
    r.synth.usesF0 = m.hasInput("pitchf");
    const TensorInfo* f = m.input(m.hasInput("phone") ? "phone" : "feats");
    if (f && f->shape.size() == 3 && f->shape[2] > 0) r.synth.channels = static_cast<int>(f->shape[2]);
    r.synth.layout = m.hasInput("phone") ? "rvc-webui" : "w-okada";
  } else if (m.hasInput("source") || m.hasInput("input_values") || m.hasOutput("unit12") || m.hasOutput("embed") ||
             m.hasOutput("hidden_states")) {
    r.kind = "encoder";
  } else if (m.hasInput("waveform") && m.hasInput("threshold")) {
    r.kind = "rmvpe";
  } else if (!r.inputs.empty() && r.inputs[0].shape.size() == 3 && !r.outputs.empty() && !r.outputs[0].shape.empty() &&
             r.outputs[0].shape.back() == 360) {
    const auto& s = r.inputs[0].shape;
    r.kind = (s[1] == 128 && s[2] != 128) ? "rmvpe" : "fcpe";
  } else {
    r.kind = "unknown";
  }
  return r;
}

std::vector<float> VoiceService::analyzePitch(const std::string& wavPath, const std::string& method,
                                              const std::string& modelPath) {
  AudioBuffer b = readWav(wavPath);
  auto x = resample(b.samples.data(), b.samples.size(), b.sampleRate, 16000);
  SessionConfig sc;
  sc.lowMemory = true;
  auto ex = createPitchExtractor(pitchMethodFromString(method), modelPath, sc, PitchOptions{});
  return ex->extract(x.data(), x.size(), x.size() / 160);
}

BenchmarkResult VoiceService::benchmark(double seconds) {
  std::lock_guard<std::mutex> lk(engineMutex_);
  if (!engine_) throw std::runtime_error("no voice loaded");
  // Synthetic vowel-like test signal: harmonic stack with vibrato.
  const size_t n = static_cast<size_t>(seconds * 16000);
  std::vector<float> x(n);
  double phase = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double t = i / 16000.0;
    const double f = 140.0 + 20.0 * std::sin(2 * kPi * 5.0 * t);
    phase += 2 * kPi * f / 16000.0;
    double s = 0;
    for (int h = 1; h <= 8; ++h) s += std::sin(h * phase) / h;
    x[i] = static_cast<float>(0.2 * s);
  }
  BenchmarkResult r;
  size_t outN = 0;
  ConversionStats st = engine_->convertBuffer(x.data(), x.size(), [&](const float*, size_t k) { outN += k; });
  r.audioSeconds = seconds;
  r.stages = st.stages;
  r.realtimeFactor = st.realtimeFactor;
  r.outputSampleRate = st.outputSampleRate;
  r.providers = engine_->info().providers;
  return r;
}

// ------------------------------------------------------------------ TTS

TtsInfo VoiceService::ttsLoad(const std::string& modelPath, const SessionConfig& cfg) {
  auto model = std::make_unique<tts::KokoroModel>(modelPath, cfg);
  std::lock_guard<std::mutex> lk(ttsMutex_);
  tts_ = std::move(model);
  return {tts_->provider(), tts_->warnings()};
}

void VoiceService::ttsUnload() {
  std::lock_guard<std::mutex> lk(ttsMutex_);
  tts_.reset();
  ttsVoice_.reset();
  ttsVoicePath_.clear();
}

bool VoiceService::ttsIsLoaded() const {
  std::lock_guard<std::mutex> lk(ttsMutex_);
  return tts_ != nullptr;
}

TtsResult VoiceService::ttsSynthesize(const std::vector<std::vector<int64_t>>& windows, const std::vector<float>& pausesAfter,
                                      const std::string& voicePath, float speed, const std::string& outWav) {
  std::lock_guard<std::mutex> lk(ttsMutex_);
  if (!tts_) throw std::runtime_error("text-to-speech model is not loaded");
  if (!ttsVoice_ || ttsVoicePath_ != voicePath) {
    ttsVoice_ = std::make_unique<tts::KokoroVoice>(tts::KokoroVoice::load(voicePath));
    ttsVoicePath_ = voicePath;
  }
  tts::SynthesisTimings t;
  const auto audio = tts::synthesize(*tts_, *ttsVoice_, windows, pausesAfter, std::clamp(speed, 0.5f, 2.0f), &t);
  if (audio.empty()) throw std::runtime_error("nothing to synthesise");
  writeWavPcm16(outWav, audio.data(), audio.size(), tts::kKokoroSampleRate);
  TtsResult r;
  r.path = outWav;
  r.sampleRate = tts::kKokoroSampleRate;
  r.seconds = static_cast<double>(audio.size()) / tts::kKokoroSampleRate;
  r.inferMs = t.inferMs;
  r.windows = t.windows;
  r.realtimeFactor = r.seconds > 0 ? (t.inferMs / 1000.0) / r.seconds : 0;
  return r;
}

}  // namespace mobigpt::rvc
