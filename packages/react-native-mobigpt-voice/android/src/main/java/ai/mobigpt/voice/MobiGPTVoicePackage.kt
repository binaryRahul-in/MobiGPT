package ai.mobigpt.voice

import com.facebook.react.BaseReactPackage
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.module.model.ReactModuleInfo
import com.facebook.react.module.model.ReactModuleInfoProvider

class MobiGPTVoicePackage : BaseReactPackage() {
  override fun getModule(name: String, reactContext: ReactApplicationContext): NativeModule? =
    if (name == MobiGPTVoiceModule.NAME) MobiGPTVoiceModule(reactContext) else null

  override fun getReactModuleInfoProvider(): ReactModuleInfoProvider = ReactModuleInfoProvider {
    mapOf(
      MobiGPTVoiceModule.NAME to ReactModuleInfo(
        MobiGPTVoiceModule.NAME,
        MobiGPTVoiceModule.NAME,
        false, // canOverrideExistingModule
        false, // needsEagerInit
        false, // isCxxModule
        true, // isTurboModule
      ),
    )
  }
}
