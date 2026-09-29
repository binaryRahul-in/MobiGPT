#pragma once

#include <jni.h>

namespace mobigpt::android {

JavaVM* javaVm();

// Returns a JNIEnv for the current thread, attaching it if needed. Threads
// attached here are detached automatically when they exit.
JNIEnv* env();

}  // namespace mobigpt::android
