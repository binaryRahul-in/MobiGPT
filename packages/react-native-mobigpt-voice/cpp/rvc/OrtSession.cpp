#include "OrtSession.h"

#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>

#if defined(__ANDROID__)
#if __has_include(<nnapi_provider_factory.h>)
#include <nnapi_provider_factory.h>
#define MOBIGPT_HAS_NNAPI 1
#endif
#endif

#if defined(__APPLE__)
#if __has_include(<coreml_provider_factory.h>)
#include <coreml_provider_factory.h>
#define MOBIGPT_HAS_COREML 1
#elif __has_include(<onnxruntime/coreml_provider_factory.h>)
#include <onnxruntime/coreml_provider_factory.h>
#define MOBIGPT_HAS_COREML 1
#endif
#endif

namespace mobigpt::rvc {

Accelerator acceleratorFromString(const std::string& s) {
  if (s == "cpu") return Accelerator::Cpu;
  if (s == "xnnpack") return Accelerator::Xnnpack;
  if (s == "nnapi") return Accelerator::Nnapi;
  if (s == "qnn" || s == "npu") return Accelerator::Qnn;
  if (s == "coreml" || s == "ane") return Accelerator::CoreMl;
  return Accelerator::Auto;
}

const char* acceleratorName(Accelerator a) {
  switch (a) {
    case Accelerator::Cpu: return "cpu";
    case Accelerator::Xnnpack: return "xnnpack";
    case Accelerator::Nnapi: return "nnapi";
    case Accelerator::Qnn: return "qnn";
    case Accelerator::CoreMl: return "coreml";
    case Accelerator::Auto: default: return "auto";
  }
}

Ort::Env& ortEnv() {
  static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "mobigpt-voice");
  return env;
}

std::vector<std::string> availableProviders() { return Ort::GetAvailableProviders(); }

std::string ortVersion() { return Ort::GetVersionString(); }

Ort::MemoryInfo& OnnxModel::cpuMemory() {
  static Ort::MemoryInfo info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  return info;
}

namespace {

bool providerCompiledIn(const char* ortName) {
  for (const auto& p : Ort::GetAvailableProviders()) {
    if (p == ortName) return true;
  }
  return false;
}

int defaultThreads() {
  const unsigned hc = std::thread::hardware_concurrency();
  if (hc == 0) return 4;
  // Leave headroom for the UI + audio threads on big.LITTLE phones.
  return static_cast<int>(hc <= 4 ? hc : hc - 2);
}

// Returns the provider name that was successfully appended ("cpu" if none).
std::string appendProvider(Ort::SessionOptions& so, Accelerator want, const SessionConfig& cfg,
                           std::vector<std::string>& warnings) {
  const int threads = cfg.intraOpThreads > 0 ? cfg.intraOpThreads : defaultThreads();
  if (want == Accelerator::Auto) {
#if defined(__ANDROID__)
    want = Accelerator::Xnnpack;
#else
    want = Accelerator::Cpu;
#endif
  }
  try {
    switch (want) {
      case Accelerator::Xnnpack:
        if (!providerCompiledIn("XnnpackExecutionProvider")) {
          warnings.push_back("XNNPACK not available in this ONNX Runtime build; using CPU");
          break;
        }
        // XNNPACK owns its own thread-pool; keep ORT's at 1 to avoid contention.
        so.SetIntraOpNumThreads(1);
        so.AppendExecutionProvider("XNNPACK", {{"intra_op_num_threads", std::to_string(threads)}});
        return "xnnpack";
      case Accelerator::Nnapi:
#if defined(MOBIGPT_HAS_NNAPI)
      {
        uint32_t flags = 0;
        if (cfg.fp16Relaxed) flags |= NNAPI_FLAG_USE_FP16;
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(so, flags));
        return "nnapi";
      }
#else
        warnings.push_back("NNAPI is Android-only; using CPU");
        break;
#endif
      case Accelerator::Qnn:
        if (!providerCompiledIn("QNNExecutionProvider")) {
          warnings.push_back("QNN (Hexagon NPU) needs the QNN build flavour (voice.ortFlavor=qnn); using CPU");
          break;
        }
        so.AppendExecutionProvider("QNN", {{"backend_path", "libQnnHtp.so"},
                                           {"htp_performance_mode", "burst"},
                                           {"enable_htp_fp16_precision", cfg.fp16Relaxed ? "1" : "0"}});
        return "qnn";
      case Accelerator::CoreMl:
#if defined(MOBIGPT_HAS_COREML)
      {
        // ML Program format supports far more ops than the legacy NeuralNetwork format.
        const uint32_t flags = COREML_FLAG_CREATE_MLPROGRAM;
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(so, flags));
        return "coreml";
      }
#else
        warnings.push_back("CoreML is Apple-only; using CPU");
        break;
#endif
      case Accelerator::Cpu:
      case Accelerator::Auto:
        break;
    }
  } catch (const Ort::Exception& e) {
    warnings.push_back(std::string(acceleratorName(want)) + " unavailable (" + e.what() + "); using CPU");
  }
  so.SetIntraOpNumThreads(threads);
  return "cpu";
}

TensorInfo describe(const Ort::TypeInfo& ti, std::string name) {
  TensorInfo info;
  info.name = std::move(name);
  if (ti.GetONNXType() == ONNX_TYPE_TENSOR) {
    auto t = ti.GetTensorTypeAndShapeInfo();
    info.type = t.GetElementType();
    info.shape = t.GetShape();
  }
  return info;
}

}  // namespace

OnnxModel::OnnxModel(const std::string& path, const SessionConfig& cfg) : path_(path) {
  auto build = [&](Accelerator acc) {
    Ort::SessionOptions so;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    so.SetInterOpNumThreads(1);
    if (cfg.lowMemory) {
      so.DisableCpuMemArena();
      so.DisableMemPattern();
    }
    provider_ = appendProvider(so, acc, cfg, warnings_);
    session_ = std::make_unique<Ort::Session>(ortEnv(), path.c_str(), so);
  };
  try {
    build(cfg.accelerator);
  } catch (const Ort::Exception& e) {
    if (provider_ == "cpu") throw std::runtime_error(std::string("failed to load ") + path + ": " + e.what());
    // Some accelerators reject a graph only at session creation; retry on CPU.
    warnings_.push_back(provider_ + " rejected the model (" + e.what() + "); retrying on CPU");
    build(Accelerator::Cpu);
  }

  Ort::AllocatorWithDefaultOptions alloc;
  for (size_t i = 0; i < session_->GetInputCount(); ++i) {
    auto n = session_->GetInputNameAllocated(i, alloc);
    inputs_.push_back(describe(session_->GetInputTypeInfo(i), n.get()));
  }
  for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
    auto n = session_->GetOutputNameAllocated(i, alloc);
    outputs_.push_back(describe(session_->GetOutputTypeInfo(i), n.get()));
  }
  auto meta = session_->GetModelMetadata();
  auto keys = meta.GetCustomMetadataMapKeysAllocated(alloc);
  for (auto& k : keys) {
    auto v = meta.LookupCustomMetadataMapAllocated(k.get(), alloc);
    if (v) metadata_[k.get()] = v.get();
  }
}

bool OnnxModel::hasInput(const std::string& name) const { return input(name) != nullptr; }
bool OnnxModel::hasOutput(const std::string& name) const { return output(name) != nullptr; }

const TensorInfo* OnnxModel::input(const std::string& name) const {
  for (const auto& i : inputs_) {
    if (i.name == name) return &i;
  }
  return nullptr;
}

const TensorInfo* OnnxModel::output(const std::string& name) const {
  for (const auto& o : outputs_) {
    if (o.name == name) return &o;
  }
  return nullptr;
}

std::vector<Ort::Value> OnnxModel::run(const std::vector<const char*>& inputNames, std::vector<Ort::Value>& inputs,
                                      const std::vector<const char*>& outputNames) {
  return session_->Run(Ort::RunOptions{nullptr}, inputNames.data(), inputs.data(), inputs.size(),
                       outputNames.data(), outputNames.size());
}

uint16_t floatToHalf(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((x >> 23) & 0xff) - 127 + 15;
  uint32_t mant = x & 0x7fffffu;
  if (((x >> 23) & 0xff) == 0xff) return static_cast<uint16_t>(sign | 0x7c00u | (mant ? 0x200u : 0u));
  if (exp <= 0) {
    if (exp < -10) return static_cast<uint16_t>(sign);
    mant |= 0x800000u;
    const uint32_t shift = static_cast<uint32_t>(14 - exp);
    uint32_t half = mant >> shift;
    if ((mant >> (shift - 1)) & 1u) half += 1;  // round half up
    return static_cast<uint16_t>(sign | half);
  }
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
  uint32_t half = sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
  if (mant & 0x1000u) half += 1;  // round to nearest
  return static_cast<uint16_t>(half);
}

float halfToFloat(uint16_t h) {
  const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x3ffu;
  uint32_t x;
  if (exp == 0) {
    if (mant == 0) {
      x = sign;
    } else {
      exp = 127 - 15 + 1;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x3ffu;
      x = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    x = sign | 0x7f800000u | (mant << 13);
  } else {
    x = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}

std::vector<float> toFloatVector(const Ort::Value& v, std::vector<int64_t>* shapeOut) {
  auto info = v.GetTensorTypeAndShapeInfo();
  const size_t n = info.GetElementCount();
  if (shapeOut) *shapeOut = info.GetShape();
  std::vector<float> out(n);
  switch (info.GetElementType()) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
      const float* p = v.GetTensorData<float>();
      std::copy(p, p + n, out.begin());
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
      const auto* p = reinterpret_cast<const uint16_t*>(v.GetTensorRawData());
      for (size_t i = 0; i < n; ++i) out[i] = halfToFloat(p[i]);
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
      const double* p = v.GetTensorData<double>();
      for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(p[i]);
      break;
    }
    default:
      throw std::runtime_error("unsupported output tensor type");
  }
  return out;
}

}  // namespace mobigpt::rvc
