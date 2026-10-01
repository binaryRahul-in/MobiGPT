require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))
ort_version = ENV["MOBIGPT_ORT_VERSION"] || "1.24.3"

Pod::Spec.new do |s|
  s.name         = "react-native-mobigpt-voice"
  s.version      = package["version"]
  s.summary      = package["description"]
  s.homepage     = "https://github.com/binaryRahul-in/MobiGPT"
  s.license      = package["license"]
  s.authors      = "MobiGPT contributors"
  s.platforms    = { :ios => "15.1" }
  s.source       = { :git => "https://github.com/binaryRahul-in/MobiGPT.git", :tag => "v#{s.version}" }

  s.source_files = [
    "ios/**/*.{h,m,mm}",
    "cpp/rvc/**/*.{h,cpp}",
    "cpp/tts/**/*.{h,cpp}",
    "cpp/jsi/**/*.{h,cpp}",
    "cpp/platform/AudioIO.h",
    "cpp/third_party/world/**/*.{h,cpp}",
  ]
  s.private_header_files = "cpp/**/*.h"
  s.frameworks = "AVFoundation", "Accelerate"
  s.libraries = "z"  # ZIP/DEFLATE for importing .pth/.zip voices

  # ONNX Runtime C/C++ API (+ CoreML execution provider) — same runtime the
  # Android build links, so the C++ engine is identical on both platforms.
  s.dependency "onnxruntime-c", ort_version

  s.pod_target_xcconfig = {
    "CLANG_CXX_LANGUAGE_STANDARD" => "c++20",
    "GCC_OPTIMIZATION_LEVEL" => "3",
    "HEADER_SEARCH_PATHS" => [
      "\"$(PODS_TARGET_SRCROOT)/cpp\"",
      "\"$(PODS_TARGET_SRCROOT)/cpp/rvc\"",
      "\"$(PODS_TARGET_SRCROOT)/cpp/platform\"",
      "\"$(PODS_TARGET_SRCROOT)/cpp/third_party/world\"",
    ].join(" "),
    "OTHER_CPLUSPLUSFLAGS" => "$(inherited) -fexceptions -frtti",
  }

  install_modules_dependencies(s)
end
