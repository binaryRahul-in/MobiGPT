// MobiGPT Voice — imports RVC voices shared as PyTorch checkpoints (.pth, or a .zip
// holding one, usually next to a FAISS .index that MobiGPT does not need) on the phone.
//
// No PyTorch is involved: the .pth is a ZIP of a pickle plus raw tensor storages. The
// pickle is read by a small interpreter that only understands what torch.save emits
// (it never executes code), and the tensors are copied as FP16 into `weights.bin`
// next to a weight-free graph template (tools/rvc/make_voice_templates.py) whose
// initializers point into that file.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mobigpt::rvc {

// ZIP reader: stored and DEFLATE entries (zlib), ZIP64 sizes.
class ZipReader {
 public:
  struct Entry {
    std::string name;
    uint64_t compressedSize = 0;
    uint64_t size = 0;
    uint64_t localHeaderOffset = 0;
    uint16_t method = 0;  // 0 = stored, 8 = deflate
  };

  explicit ZipReader(const std::string& path);
  const std::vector<Entry>& entries() const { return entries_; }
  const Entry* find(const std::string& name) const;
  std::vector<uint8_t> read(const Entry& e) const;
  void extract(const Entry& e, const std::string& destPath) const;
  const std::string& path() const { return path_; }

 private:
  uint64_t dataOffset(const Entry& e) const;
  std::string path_;
  std::vector<Entry> entries_;
};

struct RvcCheckpointInfo {
  std::string pthName;        // entry name inside the archive (or the file name)
  std::string version;        // "v1" (256-d features) or "v2" (768-d)
  int sampleRate = 0;
  bool f0 = true;             // pitch-guided
  int speakers = 0;           // rows of emb_g
  int featureDim = 0;
  std::string dtype;          // "float16" | "float32" | "bfloat16"
  size_t tensors = 0;
  bool hasIndex = false;      // a FAISS .index was bundled (ignored on device)
  std::string info;           // RVC's training note, e.g. "300epoch"
};

// Reads the checkpoint's description without touching the tensor data.
RvcCheckpointInfo inspectRvcCheckpoint(const std::string& pthOrZip);

struct RvcImportResult {
  std::string modelPath;      // outDir/model.onnx
  uint64_t weightsBytes = 0;  // outDir/weights.bin
  RvcCheckpointInfo info;
};

// Writes outDir/model.onnx (a copy of the template) and outDir/weights.bin.
RvcImportResult importRvcCheckpoint(const std::string& pthOrZip, const std::string& templatePath, const std::string& outDir);

// Template file name the app downloads for a checkpoint, e.g. "rvc_template_v2_40k.onnx".
std::string rvcTemplateName(const RvcCheckpointInfo& info);

}  // namespace mobigpt::rvc
