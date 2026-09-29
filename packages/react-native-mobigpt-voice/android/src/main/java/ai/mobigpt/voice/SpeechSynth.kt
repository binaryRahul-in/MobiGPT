package ai.mobigpt.voice

import android.content.Context
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import java.io.File
import java.io.RandomAccessFile
import java.util.Locale
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap

/**
 * Offline-first text-to-speech through the OS engine, rendered to a WAV file
 * so it can be fed straight into the RVC engine (TTS -> voice conversion).
 */
class SpeechSynth(private val context: Context) {
  data class VoiceInfo(
    val id: String,
    val name: String,
    val language: String,
    val quality: Int,
    val requiresNetwork: Boolean,
  )

  private var tts: TextToSpeech? = null
  @Volatile private var ready = false
  private val pending = mutableListOf<() -> Unit>()
  private val callbacks = ConcurrentHashMap<String, Pair<(Double, Int) -> Unit, (String) -> Unit>>()
  private val outputs = ConcurrentHashMap<String, String>()

  private fun withEngine(onError: (String) -> Unit, block: (TextToSpeech) -> Unit) {
    synchronized(this) {
      val engine = tts
      if (engine != null && ready) {
        block(engine)
        return
      }
      pending.add { tts?.let(block) ?: onError("text-to-speech engine unavailable") }
      if (engine != null) return
      tts = TextToSpeech(context.applicationContext) { status ->
        synchronized(this) {
          ready = status == TextToSpeech.SUCCESS
          val queued = pending.toList()
          pending.clear()
          if (ready) {
            tts?.setOnUtteranceProgressListener(listener)
            queued.forEach { it() }
          } else {
            tts = null
            queued.forEach { _ -> onError("no text-to-speech engine installed") }
          }
        }
      }
    }
  }

  private val listener = object : UtteranceProgressListener() {
    override fun onStart(utteranceId: String) {}

    override fun onDone(utteranceId: String) {
      val cb = callbacks.remove(utteranceId) ?: return
      val path = outputs.remove(utteranceId) ?: return
      val (rate, seconds) = readWavInfo(path)
      cb.first(seconds, rate)
    }

    @Deprecated("Deprecated in Java")
    override fun onError(utteranceId: String) {
      outputs.remove(utteranceId)
      callbacks.remove(utteranceId)?.second?.invoke("speech synthesis failed")
    }

    override fun onError(utteranceId: String, errorCode: Int) {
      outputs.remove(utteranceId)
      callbacks.remove(utteranceId)?.second?.invoke("speech synthesis failed (code $errorCode)")
    }
  }

  fun synthesize(
    text: String,
    outputPath: String,
    language: String,
    voiceId: String,
    rate: Float,
    pitch: Float,
    onDone: (Double, Int) -> Unit,
    onError: (String) -> Unit,
  ) {
    withEngine(onError) { engine ->
      try {
        if (voiceId.isNotEmpty()) {
          engine.voices?.firstOrNull { it.name == voiceId }?.let { engine.voice = it }
        } else if (language.isNotEmpty()) {
          engine.language = Locale.forLanguageTag(language)
        }
        engine.setSpeechRate(if (rate > 0f) rate else 1f)
        engine.setPitch(if (pitch > 0f) pitch else 1f)
        val id = UUID.randomUUID().toString()
        callbacks[id] = onDone to onError
        outputs[id] = outputPath
        File(outputPath).parentFile?.mkdirs()
        val result = engine.synthesizeToFile(text, Bundle(), File(outputPath), id)
        if (result != TextToSpeech.SUCCESS) {
          callbacks.remove(id)
          outputs.remove(id)
          onError("speech synthesis request rejected")
        }
      } catch (e: Throwable) {
        onError(e.message ?: "speech synthesis failed")
      }
    }
  }

  fun voices(onDone: (List<VoiceInfo>) -> Unit, onError: (String) -> Unit) {
    withEngine(onError) { engine ->
      val list = (engine.voices ?: emptySet()).map {
        VoiceInfo(it.name, it.name, it.locale.toLanguageTag(), it.quality, it.isNetworkConnectionRequired)
      }.sortedWith(compareBy({ it.requiresNetwork }, { it.language }, { -it.quality }))
      onDone(list)
    }
  }

  fun shutdown() {
    synchronized(this) {
      tts?.shutdown()
      tts = null
      ready = false
    }
  }

  private fun readWavInfo(path: String): Pair<Int, Double> {
    return try {
      RandomAccessFile(path, "r").use { f ->
        val header = ByteArray(44)
        f.readFully(header)
        fun le32(o: Int) = (header[o].toInt() and 0xff) or ((header[o + 1].toInt() and 0xff) shl 8) or
          ((header[o + 2].toInt() and 0xff) shl 16) or ((header[o + 3].toInt() and 0xff) shl 24)
        val rate = le32(24)
        val byteRate = le32(28)
        val seconds = if (byteRate > 0) (f.length() - 44).toDouble() / byteRate else 0.0
        rate to seconds
      }
    } catch (e: Throwable) {
      0 to 0.0
    }
  }
}
