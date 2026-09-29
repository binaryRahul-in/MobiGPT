// JNI entry points for MobiGPT Voice.
#include <ReactCommon/CallInvokerHolder.h>
#include <android/log.h>
#include <fbjni/fbjni.h>
#include <jni.h>
#include <jsi/jsi.h>

#include "JniEnv.h"
#include "jsi/VoiceJSI.h"

namespace mobigpt::android {

namespace {
JavaVM* gVm = nullptr;

struct Detacher {
  bool attached = false;
  ~Detacher() {
    if (attached && gVm) gVm->DetachCurrentThread();
  }
};
thread_local Detacher tDetacher;
}  // namespace

JavaVM* javaVm() { return gVm; }

JNIEnv* env() {
  JNIEnv* e = nullptr;
  if (!gVm) return nullptr;
  if (gVm->GetEnv(reinterpret_cast<void**>(&e), JNI_VERSION_1_6) == JNI_OK) return e;
  if (gVm->AttachCurrentThread(&e, nullptr) != JNI_OK) return nullptr;
  tDetacher.attached = true;
  return e;
}

}  // namespace mobigpt::android

extern "C" JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void*) {
  mobigpt::android::gVm = vm;
  return facebook::jni::initialize(vm, [] {});
}

extern "C" JNIEXPORT void JNICALL Java_ai_mobigpt_voice_MobiGPTVoiceModule_nativeInstall(JNIEnv*, jobject, jlong runtimePtr,
                                                                                         jobject callInvokerHolder) {
  if (runtimePtr == 0 || callInvokerHolder == nullptr) return;
  auto holder = facebook::jni::alias_ref<facebook::react::CallInvokerHolder::javaobject>{
      reinterpret_cast<facebook::react::CallInvokerHolder::javaobject>(callInvokerHolder)};
  auto invoker = holder->cthis()->getCallInvoker();
  if (!invoker) return;
  invoker->invokeAsync([invoker](facebook::jsi::Runtime& rt) {
    try {
      mobigpt::jsi_bindings::install(rt, invoker);
    } catch (const std::exception& e) {
      __android_log_print(ANDROID_LOG_ERROR, "MobiGPTVoice", "JSI install failed: %s", e.what());
    }
  });
}

extern "C" JNIEXPORT void JNICALL Java_ai_mobigpt_voice_MobiGPTVoiceModule_nativeInvalidate(JNIEnv*, jobject) {
  mobigpt::jsi_bindings::invalidate();
}
