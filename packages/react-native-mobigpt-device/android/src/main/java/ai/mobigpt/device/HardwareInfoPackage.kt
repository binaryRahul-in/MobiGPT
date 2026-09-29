package ai.mobigpt.device

import com.facebook.react.BaseReactPackage
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.module.model.ReactModuleInfo
import com.facebook.react.module.model.ReactModuleInfoProvider

class HardwareInfoPackage : BaseReactPackage() {
  override fun getModule(name: String, reactContext: ReactApplicationContext): NativeModule? =
    if (name == NativeHardwareInfoSpec.NAME) HardwareInfoModule(reactContext) else null

  override fun getReactModuleInfoProvider(): ReactModuleInfoProvider = ReactModuleInfoProvider {
    mapOf(
      NativeHardwareInfoSpec.NAME to
        ReactModuleInfo(NativeHardwareInfoSpec.NAME, NativeHardwareInfoSpec.NAME, false, false, false, true),
    )
  }
}
