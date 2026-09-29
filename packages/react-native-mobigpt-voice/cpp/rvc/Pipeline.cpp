#include "Pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

#include "Audio.h"
#include "Dsp.h"

namespace mobigpt::rvc {

namespace {

using Clock = std::chrono::steady_clock;
double msSince(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

int roundTo(double v, int multiple) {
  const int m = static_cast<int>(std::lround(v / multiple)) * multiple;
  return std::max(multiple, m);
}

}  // namespace

// ------------------------------------------------------------------ helpers

void mixRms(const float* ref, size_t refCount, float* out, size_t outCount, int outRate, float rate) {
  if (rate >= 1.0f || refCount == 0 || outCount == 0) return;
  auto envelope = [](const float* x, size_t n, int sr) {
    // librosa.feature.rms(frame_length=sr, hop_length=sr/2, center=True)
    const long frame = sr, hop = sr / 2;
    const long frames = 1 + static_cast<long>(n) / hop;
    std::vector<float> e(static_cast<size_t>(frames));
    for (long f = 0; f < frames; ++f) {
      const long c = f * hop;
      double s = 0.0;
      for (long i = c - frame / 2; i < c + frame / 2; ++i) {
        if (i >= 0 && i < static_cast<long>(n)) s += static_cast<double>(x[i]) * x[i];
      }
      e[static_cast<size_t>(f)] = static_cast<float>(std::sqrt(s / frame));
    }
    return e;
  };
  auto interp = [](const std::vector<float>& e, size_t n) {
    std::vector<float> y(n);
    if (e.size() == 1) {
      std::fill(y.begin(), y.end(), e[0]);
      return y;
    }
    for (size_t i = 0; i < n; ++i) {
      const double pos = static_cast<double>(i) * (e.size() - 1) / std::max<size_t>(1, n - 1);
      const size_t a = static_cast<size_t>(pos);
      const size_t b = std::min(a + 1, e.size() - 1);
      const double t = pos - a;
      y[i] = static_cast<float>(e[a] * (1 - t) + e[b] * t);
    }
    return y;
  };
  const auto r1 = interp(envelope(ref, refCount, 16000), outCount);
  const auto r2 = interp(envelope(out, outCount, outRate), outCount);
  for (size_t i = 0; i < outCount; ++i) {
    const double a = std::pow(std::max(r1[i], 1e-6f), 1.0 - rate);
    const double b = std::pow(std::max(r2[i], 1e-6f), rate - 1.0);
    out[i] = static_cast<float>(out[i] * a * b);
  }
}

// ------------------------------------------------------- StreamingConverter

StreamingConverter::Plan StreamingConverter::makePlan(const PipelineConfig& cfg) {
  Plan p;
  p.core = roundTo(std::max(0.5f, cfg.chunkSeconds) * 16000.0, 320);
  p.context = roundTo(std::max(0.05f, cfg.contextSeconds) * 16000.0, 320);
  p.xfade = roundTo(std::max(20.0f, cfg.crossfadeMs) * 16.0, 320);
  p.sola = std::max(0, static_cast<int>(std::lround(cfg.solaSearchMs * 16.0 / 160.0)) * 160);
  if (p.sola > p.context) p.sola = p.context;
  if (p.xfade >= p.core) p.xfade = std::max(320, (p.core / 2 / 320) * 320);
  return p;
}

StreamingConverter::StreamingConverter(Plan plan, int outHop, WindowFn window, EmitFn emit)
    : plan_(plan), hopOut_(outHop), window_(std::move(window)), emit_(std::move(emit)) {
  if (hopOut_ <= 0) throw std::invalid_argument("output hop must be positive");
  // Left context for the very first chunk is silence.
  buffer_.assign(static_cast<size_t>(plan_.context), 0.0f);
}

void StreamingConverter::push(const float* audio, size_t count) {
  buffer_.insert(buffer_.end(), audio, audio + count);
  totalIn_ += count;
  const size_t need = static_cast<size_t>(2 * plan_.context + plan_.core + plan_.xfade);
  while (buffer_.size() >= need) processChunk();
}

void StreamingConverter::processChunk() {
  const int X = plan_.context, C = plan_.core, F = plan_.xfade, S = plan_.sola;
  const size_t W = static_cast<size_t>(2 * X + C + F);
  std::vector<float> out = window_(buffer_.data(), W);
  const size_t hop = static_cast<size_t>(hopOut_);
  const size_t expected = (W / 160) * hop;
  out.resize(expected, 0.0f);

  const size_t Xo = static_cast<size_t>(X / 160) * hop;
  const size_t Co = static_cast<size_t>(C / 160) * hop;
  const size_t Fo = static_cast<size_t>(F / 160) * hop;
  const size_t So = static_cast<size_t>(S / 160) * hop;

  size_t k = Xo;
  if (started_ && So > 0 && !tail_.empty()) {
    // SOLA: find the offset whose head best matches the held-back tail.
    double best = -1e30;
    for (size_t cand = Xo - So; cand <= Xo + So; ++cand) {
      double num = 0.0, den = 1e-9;
      for (size_t i = 0; i < Fo; ++i) {
        num += static_cast<double>(out[cand + i]) * tail_[i];
        den += static_cast<double>(out[cand + i]) * out[cand + i];
      }
      const double score = num / std::sqrt(den);
      if (score > best) {
        best = score;
        k = cand;
      }
    }
  }

  std::vector<float> emitBuf;
  emitBuf.reserve(Co);
  if (!started_) {
    emitBuf.insert(emitBuf.end(), out.begin() + static_cast<long>(k), out.begin() + static_cast<long>(k + Co));
    started_ = true;
  } else {
    for (size_t i = 0; i < Fo; ++i) {
      const double t = (i + 0.5) / Fo;
      const double fadeIn = std::sin(0.5 * kPi * t);
      const double wIn = fadeIn * fadeIn;
      emitBuf.push_back(static_cast<float>(tail_[i] * (1.0 - wIn) + out[k + i] * wIn));
    }
    emitBuf.insert(emitBuf.end(), out.begin() + static_cast<long>(k + Fo), out.begin() + static_cast<long>(k + Co));
  }
  tail_.assign(out.begin() + static_cast<long>(k + Co), out.begin() + static_cast<long>(k + Co + Fo));

  // Never emit more audio than was pushed (matters on the final flush).
  const size_t targetTotal = (totalIn_ + 159) / 160 * hop;
  const size_t already = totalOutFrames_;
  size_t n = emitBuf.size();
  if (already + n > targetTotal) n = targetTotal > already ? targetTotal - already : 0;
  if (n) emit_(emitBuf.data(), n);
  totalOutFrames_ += n;  // counts output samples

  buffer_.erase(buffer_.begin(), buffer_.begin() + C);
  ++chunks_;
}

void StreamingConverter::flush() {
  // Real input still waiting to be emitted = buffer minus the left context.
  const size_t hop = static_cast<size_t>(hopOut_);
  const size_t targetTotal = (totalIn_ + 159) / 160 * hop;
  const size_t need = static_cast<size_t>(2 * plan_.context + plan_.core + plan_.xfade);
  while (totalOutFrames_ + (started_ ? tail_.size() : 0) < targetTotal && buffer_.size() > static_cast<size_t>(plan_.context)) {
    if (buffer_.size() < need) buffer_.resize(need, 0.0f);
    processChunk();
  }
  if (!tail_.empty() && totalOutFrames_ < targetTotal) {
    const size_t n = std::min(tail_.size(), targetTotal - totalOutFrames_);
    emit_(tail_.data(), n);
    totalOutFrames_ += n;
  }
  tail_.clear();
}

// ----------------------------------------------------------------- engine

RvcEngine::RvcEngine(PipelineConfig cfg) : cfg_(std::move(cfg)) {
  if (cfg_.indexRate != 0.0f) {
    warnings_.push_back("index_rate forced to 0: FAISS retrieval index is disabled on mobile");
    cfg_.indexRate = 0.0f;
  }
}

RvcEngine::~RvcEngine() = default;

void RvcEngine::ensureEncoder() {
  if (!encoder_) encoder_ = std::make_unique<ContentEncoder>(cfg_.encoderPath, cfg_.session);
}

void RvcEngine::ensurePitch() {
  if (!pitch_) pitch_ = createPitchExtractor(cfg_.pitchMethod, cfg_.pitchModelPath, cfg_.session, cfg_.pitch);
}

void RvcEngine::ensureSynth() {
  if (!synth_) {
    synth_ = std::make_unique<Synthesizer>(cfg_.synthPath, cfg_.session, cfg_.defaultSampleRate, cfg_.seed);
    synthInfo_ = synth_->info();
  }
}

void RvcEngine::load() {
  if (cfg_.encoderPath.empty() || cfg_.synthPath.empty()) {
    throw std::invalid_argument("encoderPath and synthPath are required");
  }
  providers_.clear();
  auto note = [&](const char* what, const OnnxModel& m) {
    providers_.push_back(std::string(what) + ":" + m.provider());
    for (const auto& w : m.warnings()) warnings_.push_back(std::string(what) + ": " + w);
  };
  // The synthesiser is always opened first: its metadata decides encoder
  // output layer / width and whether pitch is needed at all.
  ensureSynth();
  note("synth", synth_->model());
  if (cfg_.strategy == LoadStrategy::Sequential) {
    synth_.reset();  // re-opened in the last stage of each conversion
  } else {
    ensureEncoder();
    note("encoder", encoder_->model());
    if (synthInfo_.usesF0) ensurePitch();
  }
  loaded_ = true;
}

void RvcEngine::unload() {
  encoder_.reset();
  pitch_.reset();
  synth_.reset();
  loaded_ = false;
}

int RvcEngine::outputSampleRate() const { return synthInfo_.sampleRate > 0 ? synthInfo_.sampleRate : cfg_.defaultSampleRate; }

EngineInfo RvcEngine::info() const {
  EngineInfo i;
  i.sampleRate = outputSampleRate();
  i.channels = synthInfo_.channels;
  i.usesF0 = synthInfo_.usesF0;
  i.synthLayout = synthInfo_.layout;
  i.version = synthInfo_.version;
  i.pitchMethod = pitchMethodName(cfg_.pitchMethod);
  i.providers = providers_;
  i.warnings = warnings_;
  i.indexUsed = false;
  return i;
}

std::vector<float> RvcEngine::encodeFeatures(const float* audio, size_t count, size_t frames, StageTimings* t) {
  const auto t0 = Clock::now();
  ensureEncoder();
  int f = 0, c = 0;
  std::vector<float> raw = encoder_->encode(audio, count, synthInfo_.channels, synthInfo_, &f, &c);
  if (c != synthInfo_.channels) {
    throw std::runtime_error("content encoder outputs " + std::to_string(c) + "-d features but the voice expects " +
                             std::to_string(synthInfo_.channels) + "-d");
  }
  // 50 Hz -> 100 Hz nearest-neighbour (F.interpolate(scale_factor=2)), then
  // pad by repeating the last frame so frames == count / 160 exactly.
  std::vector<float> feats(frames * static_cast<size_t>(c));
  for (size_t j = 0; j < frames; ++j) {
    size_t src = std::min(j / 2, static_cast<size_t>(std::max(0, f - 1)));
    std::copy(raw.begin() + static_cast<long>(src * c), raw.begin() + static_cast<long>((src + 1) * c),
              feats.begin() + static_cast<long>(j * c));
  }
  if (t) t->encoderMs += msSince(t0);
  return feats;
}

std::vector<float> RvcEngine::extractF0(const float* audio, size_t count, size_t frames, StageTimings* t) {
  if (!synthInfo_.usesF0) return std::vector<float>(frames, 0.0f);
  const auto t0 = Clock::now();
  ensurePitch();
  std::vector<float> f0 = pitch_->extract(audio, count, frames);
  if (t) t->pitchMs += msSince(t0);
  return f0;
}

std::vector<float> RvcEngine::synthesize(const std::vector<float>& feats, std::vector<float> f0, size_t frames,
                                         StageTimings* t) {
  const auto t0 = Clock::now();
  ensureSynth();
  shiftPitch(f0, cfg_.f0UpKey);
  const auto coarse = coarsePitch(f0);
  std::vector<float> y = synth_->synthesize(feats, static_cast<int>(frames), synthInfo_.channels, coarse, f0, cfg_.speakerId);
  y.resize(frames * static_cast<size_t>(synth_->hop()), 0.0f);
  if (t) t->synthMs += msSince(t0);
  return y;
}

std::vector<float> RvcEngine::convertWindow(const float* audio, size_t count, StageTimings* t) {
  if (!loaded_) throw std::runtime_error("engine not loaded");
  std::vector<float> in(audio, audio + count);
  if (in.size() % 320) in.resize((in.size() / 320 + 1) * 320, 0.0f);
  const size_t frames = in.size() / 160;
  auto feats = encodeFeatures(in.data(), in.size(), frames, t);
  auto f0 = extractF0(in.data(), in.size(), frames, t);
  auto y = synthesize(feats, std::move(f0), frames, t);
  mixRms(in.data(), in.size(), y.data(), y.size(), outputSampleRate(), cfg_.rmsMixRate);
  y.resize((count / 160) * static_cast<size_t>(outputSampleRate() / 100));
  return y;
}

ConversionStats RvcEngine::convertBuffer(const float* audio, size_t count, const EmitFn& emit, const ProgressFn& progress,
                                         const std::atomic<bool>* cancel) {
  if (!loaded_) throw std::runtime_error("engine not loaded");
  if (cfg_.strategy == LoadStrategy::Sequential) return convertSequential(audio, count, emit, progress, cancel);

  ConversionStats st;
  const auto t0 = Clock::now();
  st.inputSeconds = count / 16000.0;
  st.outputSampleRate = outputSampleRate();
  size_t outSamples = 0;
  const auto plan = StreamingConverter::makePlan(cfg_);
  StreamingConverter sc(
      plan, outputSampleRate() / 100,
      [&](const float* w, size_t n) { return convertWindow(w, n, &st.stages); },
      [&](const float* y, size_t n) {
        outSamples += n;
        emit(y, n);
      });
  const size_t block = static_cast<size_t>(plan.core);
  for (size_t off = 0; off < count; off += block) {
    if (cancel && cancel->load()) {
      st.cancelled = true;
      break;
    }
    sc.push(audio + off, std::min(block, count - off));
    if (progress) progress(std::min(1.0, static_cast<double>(off + block) / count) * 0.98);
  }
  if (!st.cancelled) sc.flush();
  st.chunks = sc.chunksProcessed();
  st.wallMs = msSince(t0);
  st.stages.totalMs = st.wallMs;
  st.outputSeconds = static_cast<double>(outSamples) / st.outputSampleRate;
  st.realtimeFactor = st.inputSeconds > 0 ? (st.wallMs / 1000.0) / st.inputSeconds : 0;
  if (progress && !st.cancelled) progress(1.0);
  return st;
}

ConversionStats RvcEngine::convertSequential(const float* audio, size_t count, const EmitFn& emit,
                                             const ProgressFn& progress, const std::atomic<bool>* cancel) {
  ConversionStats st;
  const auto t0 = Clock::now();
  st.inputSeconds = count / 16000.0;
  const auto plan = StreamingConverter::makePlan(cfg_);
  const size_t block = static_cast<size_t>(plan.core);
  auto cancelled = [&] { return cancel && cancel->load(); };

  // Dry-run the scheduler to enumerate the exact windows the final pass
  // will request; each stage then only needs its own model in RAM.
  std::vector<std::vector<uint16_t>> featStore;  // fp16 to halve memory
  std::vector<std::vector<float>> f0Store;
  auto forEachWindow = [&](const std::function<void(const float*, size_t)>& fn) {
    StreamingConverter dry(plan, 1, [&](const float* w, size_t n) {
      fn(w, n);
      return std::vector<float>(n / 160, 0.0f);
    }, [](const float*, size_t) {});
    for (size_t off = 0; off < count && !cancelled(); off += block) dry.push(audio + off, std::min(block, count - off));
    if (!cancelled()) dry.flush();
  };

  // Stage 1 — content features.
  ensureSynth();
  synthInfo_ = synth_->info();
  st.outputSampleRate = outputSampleRate();
  synth_.reset();
  forEachWindow([&](const float* w, size_t n) {
    const size_t frames = n / 160;
    auto f = encodeFeatures(w, n, frames, &st.stages);
    std::vector<uint16_t> h(f.size());
    for (size_t i = 0; i < f.size(); ++i) h[i] = floatToHalf(f[i]);
    featStore.push_back(std::move(h));
  });
  encoder_.reset();
  if (progress) progress(0.33);

  // Stage 2 — pitch.
  forEachWindow([&](const float* w, size_t n) { f0Store.push_back(extractF0(w, n, n / 160, &st.stages)); });
  pitch_.reset();
  if (progress) progress(0.5);
  if (cancelled()) {
    st.cancelled = true;
    return st;
  }

  // Stage 3 — synthesis + stitching.
  ensureSynth();
  size_t idx = 0, outSamples = 0;
  const size_t total = featStore.size();
  StreamingConverter sc(
      plan, outputSampleRate() / 100,
      [&](const float* w, size_t n) {
        if (idx >= total) throw std::logic_error("sequential window mismatch");
        const size_t frames = n / 160;
        std::vector<float> feats(featStore[idx].size());
        for (size_t i = 0; i < feats.size(); ++i) feats[i] = halfToFloat(featStore[idx][i]);
        std::vector<uint16_t>().swap(featStore[idx]);
        auto y = synthesize(feats, std::move(f0Store[idx]), frames, &st.stages);
        mixRms(w, n, y.data(), y.size(), outputSampleRate(), cfg_.rmsMixRate);
        ++idx;
        if (progress) progress(0.5 + 0.5 * static_cast<double>(idx) / std::max<size_t>(1, total));
        return y;
      },
      [&](const float* y, size_t n) {
        outSamples += n;
        emit(y, n);
      });
  for (size_t off = 0; off < count && !cancelled(); off += block) sc.push(audio + off, std::min(block, count - off));
  if (cancelled()) {
    st.cancelled = true;
  } else {
    sc.flush();
  }
  synth_.reset();
  st.chunks = sc.chunksProcessed();
  st.wallMs = msSince(t0);
  st.stages.totalMs = st.wallMs;
  st.outputSeconds = static_cast<double>(outSamples) / st.outputSampleRate;
  st.realtimeFactor = st.inputSeconds > 0 ? (st.wallMs / 1000.0) / st.inputSeconds : 0;
  return st;
}

ConversionStats RvcEngine::convertFile(const std::string& inWav, const std::string& outWav, const ProgressFn& progress,
                                       const std::atomic<bool>* cancel) {
  AudioBuffer in = readWav(inWav);
  std::vector<float> x16 = resample(in.samples.data(), in.samples.size(), in.sampleRate, 16000);
  in.samples.clear();
  in.samples.shrink_to_fit();
  WavStreamWriter w;
  // Opened lazily: in sequential mode the true rate is known only once the
  // synthesiser has been (re)opened.
  bool opened = false;
  auto emit = [&](const float* y, size_t n) {
    if (!opened) {
      w.open(outWav, outputSampleRate());
      opened = true;
    }
    w.write(y, n);
  };
  ConversionStats st = convertBuffer(x16.data(), x16.size(), emit, progress, cancel);
  if (!opened) w.open(outWav, outputSampleRate());
  w.close();
  return st;
}

}  // namespace mobigpt::rvc
