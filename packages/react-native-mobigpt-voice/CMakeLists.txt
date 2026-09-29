# Host (Linux/macOS) build of the MobiGPT Voice C++ core: unit tests + CLI.
# The Android and iOS builds use android/CMakeLists.txt and the podspec.
#
#   cmake -S . -B build -DORT_ROOT=/path/to/onnxruntime   # has include/ + lib/
#   cmake --build build -j && ctest --test-dir build --output-on-failure
cmake_minimum_required(VERSION 3.18)
project(mobigpt_voice_core CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(ORT_ROOT "" CACHE PATH "ONNX Runtime root containing include/ and lib/")
find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS ${ORT_ROOT}/include ${ORT_ROOT}/include/onnxruntime ${ORT_ROOT}/include/onnxruntime/core/session)
find_library(ORT_LIBRARY NAMES onnxruntime libonnxruntime.so.1 HINTS ${ORT_ROOT}/lib ${ORT_ROOT})
if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIBRARY)
  message(FATAL_ERROR "ONNX Runtime not found; set -DORT_ROOT=<dir with include/ and lib/>")
endif()

include(cmake/sources.cmake)

add_library(mobigpt_rvc_core STATIC ${MOBIGPT_RVC_SOURCES} cpp/platform/NullAudio.cpp)
target_include_directories(mobigpt_rvc_core PUBLIC cpp/rvc cpp/platform cpp/third_party/world ${ORT_INCLUDE_DIR})
target_link_libraries(mobigpt_rvc_core PUBLIC ${ORT_LIBRARY})
find_package(Threads REQUIRED)
target_link_libraries(mobigpt_rvc_core PUBLIC Threads::Threads)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(mobigpt_rvc_core PRIVATE -Wall -Wextra -Wno-unused-parameter)
  # Vendored WORLD code is not ours to lint.
  set_source_files_properties(${MOBIGPT_WORLD_SOURCES} PROPERTIES COMPILE_OPTIONS "-w")
endif()

option(MOBIGPT_SANITIZE "Build with ASan/UBSan" OFF)
if(MOBIGPT_SANITIZE)
  target_compile_options(mobigpt_rvc_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(mobigpt_rvc_core PUBLIC -fsanitize=address,undefined)
endif()

add_executable(mobigpt-rvc tools/cli.cpp)
target_link_libraries(mobigpt-rvc PRIVATE mobigpt_rvc_core)

enable_testing()
add_executable(rvc_tests tests/test_main.cpp tests/test_dsp.cpp tests/test_pitch.cpp tests/test_pipeline.cpp tests/test_tts.cpp)
target_link_libraries(rvc_tests PRIVATE mobigpt_rvc_core)
target_include_directories(rvc_tests PRIVATE tests)
set(MOBIGPT_FIXTURES ${CMAKE_CURRENT_BINARY_DIR}/fixtures CACHE PATH "Generated test fixtures")
target_compile_definitions(rvc_tests PRIVATE MOBIGPT_FIXTURES="${MOBIGPT_FIXTURES}")
add_test(NAME rvc_tests COMMAND rvc_tests)
