package ai.mobigpt.voice

import android.util.Log
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.module.annotations.ReactModule
import java.util.concurrent.Executors

@ReactModule(name = MobiGPTVoiceModule.NAME)
class MobiGPTVoiceModule(context: ReactApplicationContext) : NativeMobiGPTVoiceSpec(context) {
  companion object {
    const val NAME = "MobiGPTVoice"
    private const val TAG = "MobiGPTVoice"
    @Volatile private var libraryLoaded = false

    fun loadLibrary(): Boolean {
      if (libraryLoaded) return true
      return try {
        System.loadLibrary("onnxruntime")
        System.loadLibrary("mobigpt_voice")
        libraryLoaded = true
        true
      } catch (e: Throwable) {
        Log.e(TAG, "native library unavailable", e)
        false
      }
    }
  }

  private val io = Executors.newSingleThreadExecutor()
  private val speech by lazy { SpeechSynth(reactApplicationContext) }

  override fun getName(): String = NAME

  override fun install(promise: Promise) {
    try {
      if (!loadLibrary()) {
        promise.resolve(false)
        return
      }
      val jsContext = reactApplicationContext.javaScriptContextHolder?.get() ?: 0L
      val holder = reactApplicationContext.jsCallInvokerHolder
      if (jsContext == 0L || holder == null) {
        promise.resolve(false)
        return
      }
      nativeInstall(jsContext, holder)
      promise.resolve(true)
    } catch (e: Throwable) {
      Log.e(TAG, "install failed", e)
      promise.resolve(false)
    }
  }

  override fun synthesizeSpeech(
    text: String,
    outputPath: String,
    language: String,
    voiceId: String,
    rate: Double,
    pitch: Double,
    promise: Promise,
  ) {
    speech.synthesize(text, outputPath, language, voiceId, rate.toFloat(), pitch.toFloat(), { seconds, sampleRate ->
      val map = Arguments.createMap()
      map.putString("path", outputPath)
      map.putDouble("seconds", seconds)
      map.putInt("sampleRate", sampleRate)
      promise.resolve(map)
    }, { error -> promise.reject("E_TTS", error) })
  }

  override fun listTtsVoices(promise: Promise) {
    speech.voices({ voices ->
      val arr = Arguments.createArray()
      for (v in voices) {
        val m = Arguments.createMap()
        m.putString("id", v.id)
        m.putString("name", v.name)
        m.putString("language", v.language)
        m.putInt("quality", v.quality)
        m.putBoolean("requiresNetwork", v.requiresNetwork)
        arr.pushMap(m)
      }
      promise.resolve(arr)
    }, { error -> promise.reject("E_TTS", error) })
  }

  override fun decodeAudioToWav(inputPath: String, outputPath: String, promise: Promise) {
    io.execute {
      try {
        val r = AudioDecoder.decodeToWav(reactApplicationContext, inputPath, outputPath)
        val map = Arguments.createMap()
        map.putString("path", outputPath)
        map.putDouble("seconds", r.seconds)
        map.putInt("sampleRate", r.sampleRate)
        promise.resolve(map)
      } catch (e: Throwable) {
        promise.reject("E_DECODE", e.message ?: "decode failed", e)
      }
    }
  }

  override fun invalidate() {
    if (libraryLoaded) {
      try {
        nativeInvalidate()
      } catch (_: Throwable) {
      }
    }
    speech.shutdown()
    io.shutdown()
    super.invalidate()
  }

  private external fun nativeInstall(jsContext: Long, callInvokerHolder: Any)
  private external fun nativeInvalidate()
}
