#!/usr/bin/env bash
# Builds and runs the MobiGPT voice engine's C++ tests on the host (Linux/macOS).
#   ORT_ROOT=/path/to/onnxruntime  (optional; downloaded if unset)
#   SANITIZE=1                      (optional; ASan + UBSan build)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PKG="$ROOT/packages/react-native-mobigpt-voice"
WORK="${WORK_DIR:-$ROOT/build/native}"
ORT_VERSION="$(node -p "require('$ROOT/mobigpt.features.json').voice.ortVersion")"
mkdir -p "$WORK"

if [[ -z "${ORT_ROOT:-}" ]]; then
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) pkg="onnxruntime-linux-x64-$ORT_VERSION" ;;
    Linux-aarch64) pkg="onnxruntime-linux-aarch64-$ORT_VERSION" ;;
    Darwin-*) pkg="onnxruntime-osx-universal2-$ORT_VERSION" ;;
    *) echo "unsupported host"; exit 1 ;;
  esac
  ORT_ROOT="$WORK/$pkg"
  if [[ ! -d "$ORT_ROOT" ]]; then
    echo "Downloading ONNX Runtime $ORT_VERSION ($pkg)…"
    curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VERSION/$pkg.tgz" | tar xz -C "$WORK"
  fi
fi

python3 -m pip install -q numpy onnx librosa >/dev/null 2>&1 || true
python3 "$PKG/tools/make_test_fixtures.py" "$WORK/fixtures"

BUILD="$WORK/build"
EXTRA=(-DCMAKE_BUILD_TYPE=Release)
if [[ -n "${SANITIZE:-}" ]]; then
  BUILD="$WORK/build-asan"
  EXTRA=(-DCMAKE_BUILD_TYPE=Debug -DMOBIGPT_SANITIZE=ON)
fi
cmake -S "$PKG" -B "$BUILD" -DORT_ROOT="$ORT_ROOT" -DMOBIGPT_FIXTURES="$WORK/fixtures" "${EXTRA[@]}"
cmake --build "$BUILD" -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
export LD_LIBRARY_PATH="$ORT_ROOT/lib:${LD_LIBRARY_PATH:-}" DYLD_LIBRARY_PATH="$ORT_ROOT/lib:${DYLD_LIBRARY_PATH:-}"
ASAN_OPTIONS=detect_leaks=0 "$BUILD/rvc_tests"
"$BUILD/mobigpt-rvc" info
