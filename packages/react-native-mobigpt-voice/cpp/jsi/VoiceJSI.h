// MobiGPT Voice — JSI bindings. Installs `global.__MobiGPTVoice`.
//
// Every heavy call runs on a native worker thread and resolves a JS Promise
// through the CallInvoker, so the JS thread never blocks and no tensor data
// is ever marshalled into JavaScript.
#pragma once

#include <jsi/jsi.h>

#include <memory>

namespace facebook::react {
class CallInvoker;
}

namespace mobigpt::jsi_bindings {

void install(facebook::jsi::Runtime& rt, std::shared_ptr<facebook::react::CallInvoker> invoker);
// Called when the JS runtime goes away (reload / teardown): pending callbacks
// are dropped instead of touching a dead runtime.
void invalidate();

}  // namespace mobigpt::jsi_bindings
