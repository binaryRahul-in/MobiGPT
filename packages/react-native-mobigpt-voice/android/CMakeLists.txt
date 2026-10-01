cmake_minimum_required(VERSION 3.13)
project(mobigpt_voice CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ReactAndroid REQUIRED CONFIG)
find_package(fbjni REQUIRED CONFIG)

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/sources.cmake)

set(ORT_DIR ${ORT_EXTRACT_DIR})
if(NOT EXISTS ${ORT_DIR}/headers/onnxruntime_cxx_api.h)
  message(FATAL_ERROR "ONNX Runtime headers not found in ${ORT_DIR}; the extractOrt Gradle task must run first")
endif()

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION ${ORT_DIR}/jni/${ANDROID_ABI}/libonnxruntime.so)

add_library(mobigpt_voice SHARED
  ${MOBIGPT_RVC_SOURCES}
  ${MOBIGPT_VOICE_ROOT}/cpp/jsi/VoiceJSI.cpp
  src/main/cpp/OnLoad.cpp
  src/main/cpp/AndroidAudio.cpp
)

target_include_directories(mobigpt_voice PRIVATE
  ${MOBIGPT_VOICE_ROOT}/cpp/rvc
  ${MOBIGPT_VOICE_ROOT}/cpp/platform
  ${MOBIGPT_VOICE_ROOT}/cpp/third_party/world
  ${MOBIGPT_VOICE_ROOT}/cpp
  ${ORT_DIR}/headers
)

set_source_files_properties(${MOBIGPT_WORLD_SOURCES} PROPERTIES COMPILE_OPTIONS "-w")
target_compile_options(mobigpt_voice PRIVATE -O3 -fexceptions -frtti -Wall -Wno-unused-parameter)
# 16 KB page-size alignment (required for Android 15+ devices on Play).
target_link_options(mobigpt_voice PRIVATE "-Wl,-z,max-page-size=16384")

find_library(LOG_LIB log)
target_link_libraries(mobigpt_voice
  onnxruntime
  fbjni::fbjni
  ReactAndroid::jsi
  ReactAndroid::reactnative
  ${LOG_LIB}
  android
  z
)
