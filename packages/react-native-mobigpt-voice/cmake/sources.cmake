# Single source list shared by the host, Android and (mirrored in) iOS builds.
get_filename_component(MOBIGPT_VOICE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

set(MOBIGPT_WORLD_SOURCES
  ${MOBIGPT_VOICE_ROOT}/cpp/third_party/world/common.cpp
  ${MOBIGPT_VOICE_ROOT}/cpp/third_party/world/dio.cpp
  ${MOBIGPT_VOICE_ROOT}/cpp/third_party/world/fft.cpp
  ${MOBIGPT_VOICE_ROOT}/cpp/third_party/world/harvest.cpp
  ${MOBIGPT_VOICE_ROOT}/cpp/third_party/world/matlabfunctions.cpp
  ${MOBIGPT_VOICE_ROOT}/cpp/third_party/world/stonemask.cpp
)

set(MOBIGPT_RVC_SOURCES
  ${MOBIGPT_VOICE_ROOT}/cpp/rvc/Audio.cpp
  ${MOBIGPT_VOICE_ROOT}/cpp/rvc/Checkpoint.cpp
  ${MOBIGPT_VOICE_ROOT}/cpp/rvc/Dsp.cpp
  ${MOBIGPT_VOICE_ROOT}/cpp/rvc/OrtSession.cpp
  ${MOBIGPT_VOICE_ROOT}/cpp/rvc/Pitch.cpp
  ${MOBIGPT_VOICE_ROOT}/cpp/rvc/Models.cpp
  ${MOBIGPT_VOICE_ROOT}/cpp/rvc/Pipeline.cpp
  ${MOBIGPT_VOICE_ROOT}/cpp/rvc/VoiceService.cpp
  ${MOBIGPT_VOICE_ROOT}/cpp/tts/Kokoro.cpp
  ${MOBIGPT_WORLD_SOURCES}
)
