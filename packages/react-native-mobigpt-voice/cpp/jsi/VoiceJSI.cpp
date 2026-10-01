#include "VoiceJSI.h"

#include <ReactCommon/CallInvoker.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <string>
#include <thread>

#include "../rvc/Checkpoint.h"
#include "../rvc/VoiceService.h"

using namespace facebook;
using mobigpt::rvc::VoiceService;

namespace mobigpt::jsi_bindings {

namespace {

std::shared_ptr<react::CallInvoker> gInvoker;
std::atomic<uint64_t> gGeneration{0};

// ------------------------------------------------------------- conversions

std::string str(jsi::Runtime& rt, const jsi::Object& o, const char* k, const std::string& def = "") {
  if (!o.hasProperty(rt, k)) return def;
  auto v = o.getProperty(rt, k);
  return v.isString() ? v.asString(rt).utf8(rt) : def;
}

double num(jsi::Runtime& rt, const jsi::Object& o, const char* k, double def) {
  if (!o.hasProperty(rt, k)) return def;
  auto v = o.getProperty(rt, k);
  return v.isNumber() ? v.asNumber() : def;
}

bool flag(jsi::Runtime& rt, const jsi::Object& o, const char* k, bool def) {
  if (!o.hasProperty(rt, k)) return def;
  auto v = o.getProperty(rt, k);
  return v.isBool() ? v.getBool() : def;
}

rvc::PipelineConfig parseConfig(jsi::Runtime& rt, const jsi::Object& o) {
  rvc::PipelineConfig c;
  c.encoderPath = str(rt, o, "encoderPath");
  c.synthPath = str(rt, o, "voicePath");
  c.pitchModelPath = str(rt, o, "pitchModelPath");
  c.pitchMethod = rvc::pitchMethodFromString(str(rt, o, "pitchMethod", "rmvpe"));
  c.pitch.f0Min = num(rt, o, "f0Min", 50);
  c.pitch.f0Max = num(rt, o, "f0Max", 1100);
  c.session.accelerator = rvc::acceleratorFromString(str(rt, o, "accelerator", "auto"));
  c.session.intraOpThreads = static_cast<int>(num(rt, o, "threads", 0));
  c.session.lowMemory = flag(rt, o, "lowMemory", false);
  c.session.fp16Relaxed = flag(rt, o, "fp16", false);
  c.strategy = str(rt, o, "loadStrategy", "resident") == "sequential" ? rvc::LoadStrategy::Sequential
                                                                       : rvc::LoadStrategy::Resident;
  c.f0UpKey = static_cast<float>(num(rt, o, "pitchShift", 0));
  c.speakerId = static_cast<int>(num(rt, o, "speakerId", 0));
  c.rmsMixRate = static_cast<float>(num(rt, o, "rmsMixRate", 0.25));
  c.indexRate = static_cast<float>(num(rt, o, "indexRate", 0));  // forced to 0 by the engine
  c.chunkSeconds = static_cast<float>(num(rt, o, "chunkSeconds", 2.5));
  c.contextSeconds = static_cast<float>(num(rt, o, "contextSeconds", 0.3));
  c.crossfadeMs = static_cast<float>(num(rt, o, "crossfadeMs", 60));
  c.solaSearchMs = static_cast<float>(num(rt, o, "solaSearchMs", 10));
  c.defaultSampleRate = static_cast<int>(num(rt, o, "defaultSampleRate", 40000));
  return c;
}

jsi::Array strings(jsi::Runtime& rt, const std::vector<std::string>& v) {
  jsi::Array a(rt, v.size());
  for (size_t i = 0; i < v.size(); ++i) a.setValueAtIndex(rt, i, jsi::String::createFromUtf8(rt, v[i]));
  return a;
}

jsi::Object toJs(jsi::Runtime& rt, const rvc::EngineInfo& i) {
  jsi::Object o(rt);
  o.setProperty(rt, "sampleRate", i.sampleRate);
  o.setProperty(rt, "channels", i.channels);
  o.setProperty(rt, "usesF0", i.usesF0);
  o.setProperty(rt, "layout", jsi::String::createFromUtf8(rt, i.synthLayout));
  o.setProperty(rt, "version", jsi::String::createFromUtf8(rt, i.version));
  o.setProperty(rt, "pitchMethod", jsi::String::createFromUtf8(rt, i.pitchMethod));
  o.setProperty(rt, "providers", strings(rt, i.providers));
  o.setProperty(rt, "warnings", strings(rt, i.warnings));
  o.setProperty(rt, "indexUsed", i.indexUsed);
  return o;
}

jsi::Object toJs(jsi::Runtime& rt, const rvc::StageTimings& t) {
  jsi::Object o(rt);
  o.setProperty(rt, "encoderMs", t.encoderMs);
  o.setProperty(rt, "pitchMs", t.pitchMs);
  o.setProperty(rt, "synthMs", t.synthMs);
  o.setProperty(rt, "totalMs", t.totalMs);
  return o;
}

jsi::Object toJs(jsi::Runtime& rt, const rvc::ConversionStats& s) {
  jsi::Object o(rt);
  o.setProperty(rt, "inputSeconds", s.inputSeconds);
  o.setProperty(rt, "outputSeconds", s.outputSeconds);
  o.setProperty(rt, "wallMs", s.wallMs);
  o.setProperty(rt, "realtimeFactor", s.realtimeFactor);
  o.setProperty(rt, "chunks", s.chunks);
  o.setProperty(rt, "sampleRate", s.outputSampleRate);
  o.setProperty(rt, "cancelled", s.cancelled);
  o.setProperty(rt, "stages", toJs(rt, s.stages));
  return o;
}

jsi::Object toJs(jsi::Runtime& rt, const rvc::LiveStats& s) {
  jsi::Object o(rt);
  o.setProperty(rt, "running", s.running);
  o.setProperty(rt, "latencyMs", s.latencyMs);
  o.setProperty(rt, "lastChunkMs", s.lastChunkMs);
  o.setProperty(rt, "realtimeFactor", s.realtimeFactor);
  o.setProperty(rt, "chunks", s.chunks);
  o.setProperty(rt, "droppedBlocks", s.droppedBlocks);
  o.setProperty(rt, "inputLevel", s.inputLevel);
  o.setProperty(rt, "outputLevel", s.outputLevel);
  o.setProperty(rt, "error", jsi::String::createFromUtf8(rt, s.error));
  return o;
}

const char* typeName(ONNXTensorElementDataType t) {
  switch (t) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "float32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "float16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "int64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return "int32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return "uint8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return "int8";
    default: return "other";
  }
}

jsi::Array tensorList(jsi::Runtime& rt, const std::vector<rvc::TensorInfo>& v) {
  jsi::Array a(rt, v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    jsi::Object t(rt);
    t.setProperty(rt, "name", jsi::String::createFromUtf8(rt, v[i].name));
    t.setProperty(rt, "type", jsi::String::createFromUtf8(rt, typeName(v[i].type)));
    jsi::Array shape(rt, v[i].shape.size());
    for (size_t j = 0; j < v[i].shape.size(); ++j) shape.setValueAtIndex(rt, j, static_cast<double>(v[i].shape[j]));
    t.setProperty(rt, "shape", shape);
    a.setValueAtIndex(rt, i, t);
  }
  return a;
}

jsi::Object toJs(jsi::Runtime& rt, const rvc::ModelInspection& m) {
  jsi::Object o(rt);
  o.setProperty(rt, "kind", jsi::String::createFromUtf8(rt, m.kind));
  o.setProperty(rt, "sizeMB", m.fileMB);
  o.setProperty(rt, "quantized", m.quantized);
  o.setProperty(rt, "inputs", tensorList(rt, m.inputs));
  o.setProperty(rt, "outputs", tensorList(rt, m.outputs));
  jsi::Object meta(rt);
  for (auto& kv : m.metadata) meta.setProperty(rt, kv.first.c_str(), jsi::String::createFromUtf8(rt, kv.second));
  o.setProperty(rt, "metadata", meta);
  if (m.kind == "synthesizer") {
    jsi::Object s(rt);
    s.setProperty(rt, "sampleRate", m.synth.sampleRate);
    s.setProperty(rt, "channels", m.synth.channels);
    s.setProperty(rt, "usesF0", m.synth.usesF0);
    s.setProperty(rt, "layout", jsi::String::createFromUtf8(rt, m.synth.layout));
    s.setProperty(rt, "version", jsi::String::createFromUtf8(rt, m.synth.channels == 256 ? "v1" : "v2"));
    o.setProperty(rt, "voice", s);
  }
  return o;
}

// ------------------------------------------------------------- promises

// Result producer runs on a worker thread; its output is converted to a
// jsi::Value on the JS thread.
using Converter = std::function<jsi::Value(jsi::Runtime&)>;

jsi::Value makePromise(jsi::Runtime& rt, std::function<Converter()> work) {
  auto promiseCtor = rt.global().getPropertyAsFunction(rt, "Promise");
  auto executor = jsi::Function::createFromHostFunction(
      rt, jsi::PropNameID::forAscii(rt, "executor"), 2,
      [work = std::move(work)](jsi::Runtime& rt, const jsi::Value&, const jsi::Value* args, size_t) -> jsi::Value {
        auto resolve = std::make_shared<jsi::Function>(args[0].asObject(rt).asFunction(rt));
        auto reject = std::make_shared<jsi::Function>(args[1].asObject(rt).asFunction(rt));
        const uint64_t gen = gGeneration.load();
        auto invoker = gInvoker;
        std::thread([work, resolve, reject, gen, invoker]() mutable {
          Converter ok;
          std::string err;
          try {
            ok = work();
          } catch (const std::exception& e) {
            err = e.what();
          } catch (...) {
            err = "unknown native error";
          }
          if (!invoker) return;
          invoker->invokeAsync([ok = std::move(ok), err, resolve, reject, gen](jsi::Runtime& rt) mutable {
            if (gen != gGeneration.load()) return;  // runtime was reloaded
            if (err.empty()) {
              resolve->call(rt, ok(rt));
            } else {
              auto errorCtor = rt.global().getPropertyAsFunction(rt, "Error");
              reject->call(rt, errorCtor.callAsConstructor(rt, jsi::String::createFromUtf8(rt, err)));
            }
            resolve.reset();
            reject.reset();
          });
        }).detach();
        return jsi::Value::undefined();
      });
  return promiseCtor.callAsConstructor(rt, executor);
}

Converter undefinedResult() {
  return [](jsi::Runtime&) { return jsi::Value::undefined(); };
}

std::string argString(jsi::Runtime& rt, const jsi::Value* args, size_t count, size_t i, const char* what) {
  if (i >= count || !args[i].isString()) throw jsi::JSError(rt, std::string("expected string argument: ") + what);
  return args[i].asString(rt).utf8(rt);
}

void setFn(jsi::Runtime& rt, jsi::Object& obj, const char* name, unsigned params, jsi::HostFunctionType fn) {
  obj.setProperty(rt, name, jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), params, std::move(fn)));
}

}  // namespace

void invalidate() {
  ++gGeneration;
  gInvoker.reset();
}

void install(jsi::Runtime& rt, std::shared_ptr<react::CallInvoker> invoker) {
  ++gGeneration;
  gInvoker = std::move(invoker);
  jsi::Object api(rt);
  auto& svc = VoiceService::instance();

  setFn(rt, api, "version", 0, [](jsi::Runtime& rt, const jsi::Value&, const jsi::Value*, size_t) -> jsi::Value {
    jsi::Object o(rt);
    o.setProperty(rt, "onnxruntime", jsi::String::createFromUtf8(rt, rvc::ortVersion()));
    o.setProperty(rt, "providers", strings(rt, rvc::availableProviders()));
    o.setProperty(rt, "engine", jsi::String::createFromUtf8(rt, "mobigpt-rvc/1"));
    return o;
  });

  setFn(rt, api, "load", 1, [&svc](jsi::Runtime& rt, const jsi::Value&, const jsi::Value* args, size_t n) -> jsi::Value {
    if (n < 1 || !args[0].isObject()) throw jsi::JSError(rt, "load(config) expects an object");
    auto cfg = parseConfig(rt, args[0].asObject(rt));
    return makePromise(rt, [cfg, &svc]() -> Converter {
      auto info = svc.load(cfg);
      return [info](jsi::Runtime& rt) { return jsi::Value(toJs(rt, info)); };
    });
  });

  setFn(rt, api, "unload", 0, [&svc](jsi::Runtime& rt, const jsi::Value&, const jsi::Value*, size_t) -> jsi::Value {
    return makePromise(rt, [&svc]() -> Converter {
      svc.unload();
      return undefinedResult();
    });
  });

  setFn(rt, api, "isLoaded", 0, [&svc](jsi::Runtime&, const jsi::Value&, const jsi::Value*, size_t) -> jsi::Value {
    return jsi::Value(svc.isLoaded());
  });

  setFn(rt, api, "setPitchShift", 1, [&svc](jsi::Runtime&, const jsi::Value&, const jsi::Value* args, size_t n) -> jsi::Value {
    if (n > 0 && args[0].isNumber()) svc.setPitchShift(static_cast<float>(args[0].asNumber()));
    return jsi::Value::undefined();
  });

  setFn(rt, api, "convertFile", 3, [&svc](jsi::Runtime& rt, const jsi::Value&, const jsi::Value* args, size_t n) -> jsi::Value {
    const std::string in = argString(rt, args, n, 0, "inputPath");
    const std::string out = argString(rt, args, n, 1, "outputPath");
    std::shared_ptr<jsi::Function> onProgress;
    if (n > 2 && args[2].isObject() && args[2].asObject(rt).isFunction(rt)) {
      onProgress = std::make_shared<jsi::Function>(args[2].asObject(rt).asFunction(rt));
    }
    return makePromise(rt, [in, out, onProgress, &svc]() -> Converter {
      auto invoker = gInvoker;
      const uint64_t gen = gGeneration.load();
      auto last = std::make_shared<std::chrono::steady_clock::time_point>();
      auto st = svc.convertFile(in, out, [onProgress, invoker, gen, last](double p) {
        if (!onProgress || !invoker) return;
        const auto now = std::chrono::steady_clock::now();
        if (p < 1.0 && now - *last < std::chrono::milliseconds(100)) return;  // throttle to 10 Hz
        *last = now;
        invoker->invokeAsync([onProgress, p, gen](jsi::Runtime& rt) {
          if (gen == gGeneration.load()) onProgress->call(rt, p);
        });
      });
      // Release the JS callback on the JS thread.
      if (invoker && onProgress) invoker->invokeAsync([onProgress](jsi::Runtime&) mutable { onProgress.reset(); });
      return [st](jsi::Runtime& rt) { return jsi::Value(toJs(rt, st)); };
    });
  });

  setFn(rt, api, "cancel", 0, [&svc](jsi::Runtime&, const jsi::Value&, const jsi::Value*, size_t) -> jsi::Value {
    svc.cancel();
    return jsi::Value::undefined();
  });

  setFn(rt, api, "startLive", 0, [&svc](jsi::Runtime& rt, const jsi::Value&, const jsi::Value*, size_t) -> jsi::Value {
    return makePromise(rt, [&svc]() -> Converter {
      svc.startLive();
      return undefinedResult();
    });
  });

  setFn(rt, api, "stopLive", 0, [&svc](jsi::Runtime& rt, const jsi::Value&, const jsi::Value*, size_t) -> jsi::Value {
    return makePromise(rt, [&svc]() -> Converter {
      svc.stopLive();
      return undefinedResult();
    });
  });

  setFn(rt, api, "liveStats", 0, [&svc](jsi::Runtime& rt, const jsi::Value&, const jsi::Value*, size_t) -> jsi::Value {
    return jsi::Value(toJs(rt, svc.liveStats()));
  });

  setFn(rt, api, "startRecording", 1, [&svc](jsi::Runtime& rt, const jsi::Value&, const jsi::Value* args, size_t n) -> jsi::Value {
    const std::string path = argString(rt, args, n, 0, "path");
    return makePromise(rt, [path, &svc]() -> Converter {
      svc.startRecording(path);
      return undefinedResult();
    });
  });

  setFn(rt, api, "stopRecording", 0, [&svc](jsi::Runtime& rt, const jsi::Value&, const jsi::Value*, size_t) -> jsi::Value {
    return makePromise(rt, [&svc]() -> Converter {
      auto r = svc.stopRecording();
      return [r](jsi::Runtime& rt) {
        jsi::Object o(rt);
        o.setProperty(rt, "path", jsi::String::createFromUtf8(rt, r.path));
        o.setProperty(rt, "seconds", r.seconds);
        o.setProperty(rt, "sampleRate", r.sampleRate);
        o.setProperty(rt, "peak", r.peak);
        return jsi::Value(std::move(o));
      };
    });
  });

  setFn(rt, api, "recordingLevel", 0, [&svc](jsi::Runtime&, const jsi::Value&, const jsi::Value*, size_t) -> jsi::Value {
    return jsi::Value(static_cast<double>(svc.recordingLevel()));
  });

  setFn(rt, api, "play", 1, [&svc](jsi::Runtime& rt, const jsi::Value&, const jsi::Value* args, size_t n) -> jsi::Value {
    const std::string path = argString(rt, args, n, 0, "path");
    return makePromise(rt, [path, &svc]() -> Converter {
      // Block this worker until playback ends so the promise resolves then.
      auto done = std::make_shared<std::promise<std::pair<bool, std::string>>>();
      auto fut = done->get_future();
      svc.play(path, [done](bool completed, std::string err) { done->set_value({completed, err}); });
      auto res = fut.get();
      return [res](jsi::Runtime& rt) {
        jsi::Object o(rt);
        o.setProperty(rt, "completed", res.first);
        return jsi::Value(std::move(o));
      };
    });
  });

  setFn(rt, api, "stopPlayback", 0, [&svc](jsi::Runtime&, const jsi::Value&, const jsi::Value*, size_t) -> jsi::Value {
    std::thread([&svc] { svc.stopPlayback(); }).detach();
    return jsi::Value::undefined();
  });

  setFn(rt, api, "isPlaying", 0, [&svc](jsi::Runtime&, const jsi::Value&, const jsi::Value*, size_t) -> jsi::Value {
    return jsi::Value(svc.isPlaying());
  });

  setFn(rt, api, "inspect", 1, [&svc](jsi::Runtime& rt, const jsi::Value&, const jsi::Value* args, size_t n) -> jsi::Value {
    const std::string path = argString(rt, args, n, 0, "path");
    return makePromise(rt, [path, &svc]() -> Converter {
      auto m = svc.inspect(path);
      return [m](jsi::Runtime& rt) { return jsi::Value(toJs(rt, m)); };
    });
  });

  setFn(rt, api, "analyzePitch", 3, [&svc](jsi::Runtime& rt, const jsi::Value&, const jsi::Value* args, size_t n) -> jsi::Value {
    const std::string path = argString(rt, args, n, 0, "path");
    const std::string method = argString(rt, args, n, 1, "method");
    const std::string model = n > 2 && args[2].isString() ? args[2].asString(rt).utf8(rt) : "";
    return makePromise(rt, [path, method, model, &svc]() -> Converter {
      auto f0 = svc.analyzePitch(path, method, model);
      return [f0](jsi::Runtime& rt) {
        jsi::Array a(rt, f0.size());
        for (size_t i = 0; i < f0.size(); ++i) a.setValueAtIndex(rt, i, static_cast<double>(f0[i]));
        return jsi::Value(std::move(a));
      };
    });
  });

  setFn(rt, api, "benchmark", 1, [&svc](jsi::Runtime& rt, const jsi::Value&, const jsi::Value* args, size_t n) -> jsi::Value {
    const double secs = n > 0 && args[0].isNumber() ? args[0].asNumber() : 5.0;
    return makePromise(rt, [secs, &svc]() -> Converter {
      auto r = svc.benchmark(secs);
      return [r](jsi::Runtime& rt) {
        jsi::Object o(rt);
        o.setProperty(rt, "audioSeconds", r.audioSeconds);
        o.setProperty(rt, "realtimeFactor", r.realtimeFactor);
        o.setProperty(rt, "sampleRate", r.outputSampleRate);
        o.setProperty(rt, "stages", toJs(rt, r.stages));
        o.setProperty(rt, "providers", strings(rt, r.providers));
        return jsi::Value(std::move(o));
      };
    });
  });

  // ------------------------------------------------------- .pth/.zip voices

  auto rvcInfoJs = [](jsi::Runtime& rt, const rvc::RvcCheckpointInfo& in) {
    jsi::Object o(rt);
    o.setProperty(rt, "pthName", jsi::String::createFromUtf8(rt, in.pthName));
    o.setProperty(rt, "version", jsi::String::createFromUtf8(rt, in.version));
    o.setProperty(rt, "sampleRate", in.sampleRate);
    o.setProperty(rt, "f0", in.f0);
    o.setProperty(rt, "speakers", in.speakers);
    o.setProperty(rt, "featureDim", in.featureDim);
    o.setProperty(rt, "dtype", jsi::String::createFromUtf8(rt, in.dtype));
    o.setProperty(rt, "tensors", static_cast<double>(in.tensors));
    o.setProperty(rt, "hasIndex", in.hasIndex);
    o.setProperty(rt, "info", jsi::String::createFromUtf8(rt, in.info));
    o.setProperty(rt, "template", jsi::String::createFromUtf8(rt, rvc::rvcTemplateName(in)));
    return o;
  };

  setFn(rt, api, "rvcInfo", 1, [rvcInfoJs](jsi::Runtime& rt, const jsi::Value&, const jsi::Value* args, size_t n) -> jsi::Value {
    const std::string path = argString(rt, args, n, 0, "path");
    return makePromise(rt, [path, rvcInfoJs]() -> Converter {
      auto in = rvc::inspectRvcCheckpoint(path);
      return [in, rvcInfoJs](jsi::Runtime& rt) { return jsi::Value(rvcInfoJs(rt, in)); };
    });
  });

  setFn(rt, api, "rvcImport", 3, [rvcInfoJs](jsi::Runtime& rt, const jsi::Value&, const jsi::Value* args, size_t n) -> jsi::Value {
    const std::string path = argString(rt, args, n, 0, "path");
    const std::string tpl = argString(rt, args, n, 1, "templatePath");
    const std::string outDir = argString(rt, args, n, 2, "outDir");
    return makePromise(rt, [path, tpl, outDir, rvcInfoJs]() -> Converter {
      auto r = rvc::importRvcCheckpoint(path, tpl, outDir);
      return [r, rvcInfoJs](jsi::Runtime& rt) {
        jsi::Object o(rt);
        o.setProperty(rt, "modelPath", jsi::String::createFromUtf8(rt, r.modelPath));
        o.setProperty(rt, "weightsBytes", static_cast<double>(r.weightsBytes));
        o.setProperty(rt, "info", rvcInfoJs(rt, r.info));
        return jsi::Value(std::move(o));
      };
    });
  });

  // ------------------------------------------------------------ neural TTS

  setFn(rt, api, "ttsLoad", 2, [&svc](jsi::Runtime& rt, const jsi::Value&, const jsi::Value* args, size_t n) -> jsi::Value {
    const std::string model = argString(rt, args, n, 0, "modelPath");
    rvc::SessionConfig sc;
    if (n > 1 && args[1].isObject()) {
      auto o = args[1].asObject(rt);
      sc.accelerator = rvc::acceleratorFromString(str(rt, o, "accelerator", "auto"));
      sc.intraOpThreads = static_cast<int>(num(rt, o, "threads", 0));
      sc.lowMemory = flag(rt, o, "lowMemory", false);
    }
    return makePromise(rt, [model, sc, &svc]() -> Converter {
      auto info = svc.ttsLoad(model, sc);
      return [info](jsi::Runtime& rt) {
        jsi::Object o(rt);
        o.setProperty(rt, "provider", jsi::String::createFromUtf8(rt, info.provider));
        o.setProperty(rt, "warnings", strings(rt, info.warnings));
        return jsi::Value(std::move(o));
      };
    });
  });

  setFn(rt, api, "ttsUnload", 0, [&svc](jsi::Runtime& rt, const jsi::Value&, const jsi::Value*, size_t) -> jsi::Value {
    return makePromise(rt, [&svc]() -> Converter {
      svc.ttsUnload();
      return undefinedResult();
    });
  });

  setFn(rt, api, "ttsIsLoaded", 0, [&svc](jsi::Runtime&, const jsi::Value&, const jsi::Value*, size_t) -> jsi::Value {
    return jsi::Value(svc.ttsIsLoaded());
  });

  // ttsSynthesize({windows: number[][], pauses: number[], voicePath, speed, outputPath})
  setFn(rt, api, "ttsSynthesize", 1, [&svc](jsi::Runtime& rt, const jsi::Value&, const jsi::Value* args, size_t n) -> jsi::Value {
    if (n < 1 || !args[0].isObject()) throw jsi::JSError(rt, "ttsSynthesize(request) expects an object");
    auto o = args[0].asObject(rt);
    std::vector<std::vector<int64_t>> windows;
    if (o.hasProperty(rt, "windows")) {
      auto arr = o.getProperty(rt, "windows").asObject(rt).asArray(rt);
      for (size_t i = 0; i < arr.size(rt); ++i) {
        auto w = arr.getValueAtIndex(rt, i).asObject(rt).asArray(rt);
        std::vector<int64_t> ids(w.size(rt));
        for (size_t j = 0; j < ids.size(); ++j) ids[j] = static_cast<int64_t>(w.getValueAtIndex(rt, j).asNumber());
        windows.push_back(std::move(ids));
      }
    }
    std::vector<float> pauses;
    if (o.hasProperty(rt, "pauses")) {
      auto arr = o.getProperty(rt, "pauses").asObject(rt).asArray(rt);
      for (size_t i = 0; i < arr.size(rt); ++i) pauses.push_back(static_cast<float>(arr.getValueAtIndex(rt, i).asNumber()));
    }
    const std::string voice = str(rt, o, "voicePath");
    const std::string out = str(rt, o, "outputPath");
    const float speed = static_cast<float>(num(rt, o, "speed", 1.0));
    return makePromise(rt, [windows = std::move(windows), pauses = std::move(pauses), voice, out, speed, &svc]() -> Converter {
      auto r = svc.ttsSynthesize(windows, pauses, voice, speed, out);
      return [r](jsi::Runtime& rt) {
        jsi::Object o(rt);
        o.setProperty(rt, "path", jsi::String::createFromUtf8(rt, r.path));
        o.setProperty(rt, "seconds", r.seconds);
        o.setProperty(rt, "sampleRate", r.sampleRate);
        o.setProperty(rt, "inferMs", r.inferMs);
        o.setProperty(rt, "realtimeFactor", r.realtimeFactor);
        o.setProperty(rt, "windows", static_cast<double>(r.windows));
        return jsi::Value(std::move(o));
      };
    });
  });

  rt.global().setProperty(rt, "__MobiGPTVoice", api);
}

}  // namespace mobigpt::jsi_bindings
