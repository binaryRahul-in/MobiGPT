// MobiGPT Voice — thin RAII wrapper over the ONNX Runtime C++ API with
// execution-provider selection and graceful CPU fallback.
#pragma once

#if __has_include(<onnxruntime_cxx_api.h>)
#include <onnxruntime_cxx_api.h>
#elif __has_include(<onnxruntime/onnxruntime_cxx_api.h>)
#include <onnxruntime/onnxruntime_cxx_api.h>
#else
#include "onnxruntime_cxx_api.h"
#endif

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mobigpt::rvc {

// Accelerator preference. Anything a platform cannot honour silently falls
// back to the next best provider; the provider actually used is reported.
enum class Accelerator {
  Auto,     // NNAPI/CoreML when likely beneficial, else XNNPACK, else CPU
  Cpu,      // default ORT CPU kernels (MLAS)
  Xnnpack,  // ARM/x86 optimised kernels, good for fp32 conv-heavy graphs
  Nnapi,    // Android GPU/DSP/NPU via NNAPI (deprecated in Android 15, still works)
  Qnn,      // Qualcomm Hexagon NPU (requires the QNN build flavour)
  CoreMl,   // Apple Neural Engine / GPU
};

Accelerator acceleratorFromString(const std::string& s);
const char* acceleratorName(Accelerator a);

struct SessionConfig {
  Accelerator accelerator = Accelerator::Auto;
  int intraOpThreads = 0;  // 0 = ORT default (all big cores)
  // Low-memory mode: disables the CPU arena + memory patterns so ORT returns
  // intermediate buffers to the OS after every Run(). ~5-15 % slower,
  // noticeably lower peak RSS on 3-4 GB phones.
  bool lowMemory = false;
  bool fp16Relaxed = false;  // allow NNAPI/CoreML to compute in fp16
};

struct TensorInfo {
  std::string name;
  ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  std::vector<int64_t> shape;  // -1 for dynamic dims
};

Ort::Env& ortEnv();
std::vector<std::string> availableProviders();
std::string ortVersion();

class OnnxModel {
 public:
  OnnxModel(const std::string& path, const SessionConfig& cfg);

  const std::vector<TensorInfo>& inputs() const { return inputs_; }
  const std::vector<TensorInfo>& outputs() const { return outputs_; }
  const std::map<std::string, std::string>& metadata() const { return metadata_; }
  const std::string& provider() const { return provider_; }
  const std::string& path() const { return path_; }
  const std::vector<std::string>& warnings() const { return warnings_; }

  bool hasInput(const std::string& name) const;
  bool hasOutput(const std::string& name) const;
  const TensorInfo* input(const std::string& name) const;
  const TensorInfo* output(const std::string& name) const;

  std::vector<Ort::Value> run(const std::vector<const char*>& inputNames, std::vector<Ort::Value>& inputs,
                              const std::vector<const char*>& outputNames);

  static Ort::MemoryInfo& cpuMemory();

 private:
  std::string path_;
  std::unique_ptr<Ort::Session> session_;
  std::vector<TensorInfo> inputs_, outputs_;
  std::map<std::string, std::string> metadata_;
  std::string provider_ = "cpu";
  std::vector<std::string> warnings_;
};

// Helpers to build tensors that *borrow* caller-owned buffers (no copies).
template <typename T>
Ort::Value borrowTensor(T* data, size_t count, const std::vector<int64_t>& shape) {
  return Ort::Value::CreateTensor<T>(OnnxModel::cpuMemory(), data, count, shape.data(), shape.size());
}

// Copies an output tensor (float or float16) into a float vector.
std::vector<float> toFloatVector(const Ort::Value& v, std::vector<int64_t>* shapeOut = nullptr);

// fp32 <-> fp16 helpers for models exported with half-precision I/O.
uint16_t floatToHalf(float f);
float halfToFloat(uint16_t h);

}  // namespace mobigpt::rvc
