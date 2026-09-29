package ai.mobigpt.voice

import android.content.Context
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Decodes any platform-supported audio (mp3/aac/m4a/ogg/opus/flac/wav) to 16-bit mono WAV. */
object AudioDecoder {
  data class Result(val sampleRate: Int, val seconds: Double)

  private const val MAX_SECONDS = 15 * 60

  fun decodeToWav(context: Context, input: String, output: String): Result {
    val extractor = MediaExtractor()
    try {
      if (input.startsWith("content://") || input.startsWith("file://")) {
        extractor.setDataSource(context, Uri.parse(input), null)
      } else {
        extractor.setDataSource(input)
      }
      var track = -1
      var format: MediaFormat? = null
      for (i in 0 until extractor.trackCount) {
        val f = extractor.getTrackFormat(i)
        if (f.getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true) {
          track = i
          format = f
          break
        }
      }
      require(track >= 0 && format != null) { "no audio track found" }
      extractor.selectTrack(track)
      val mime = format.getString(MediaFormat.KEY_MIME)!!
      var sampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
      var channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
      var floatPcm = false

      val codec = MediaCodec.createDecoderByType(mime)
      codec.configure(format, null, null, 0)
      codec.start()
      File(output).parentFile?.mkdirs()
      val out = FileOutputStream(output)
      out.write(ByteArray(44)) // header patched below
      var frames = 0L
      val info = MediaCodec.BufferInfo()
      var inputDone = false
      var outputDone = false
      try {
        while (!outputDone) {
          if (!inputDone) {
            val inIdx = codec.dequeueInputBuffer(10_000)
            if (inIdx >= 0) {
              val buf = codec.getInputBuffer(inIdx)!!
              val n = extractor.readSampleData(buf, 0)
              if (n < 0) {
                codec.queueInputBuffer(inIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                inputDone = true
              } else {
                codec.queueInputBuffer(inIdx, 0, n, extractor.sampleTime, 0)
                extractor.advance()
              }
            }
          }
          val outIdx = codec.dequeueOutputBuffer(info, 10_000)
          when {
            outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
              val f = codec.outputFormat
              sampleRate = f.getInteger(MediaFormat.KEY_SAMPLE_RATE)
              channels = f.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
              floatPcm = f.containsKey(MediaFormat.KEY_PCM_ENCODING) &&
                f.getInteger(MediaFormat.KEY_PCM_ENCODING) == AudioFormat.ENCODING_PCM_FLOAT
            }
            outIdx >= 0 -> {
              val buf = codec.getOutputBuffer(outIdx)!!
              buf.position(info.offset)
              buf.limit(info.offset + info.size)
              frames += writeMono(buf.slice().order(ByteOrder.LITTLE_ENDIAN), channels, floatPcm, out)
              codec.releaseOutputBuffer(outIdx, false)
              if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) outputDone = true
              require(frames < MAX_SECONDS.toLong() * sampleRate) { "audio longer than 15 minutes" }
            }
          }
        }
      } finally {
        codec.stop()
        codec.release()
        out.close()
      }
      patchHeader(output, sampleRate, frames)
      return Result(sampleRate, frames.toDouble() / sampleRate)
    } finally {
      extractor.release()
    }
  }

  private fun writeMono(buf: ByteBuffer, channels: Int, floatPcm: Boolean, out: FileOutputStream): Long {
    val bytesPerSample = if (floatPcm) 4 else 2
    val frameCount = buf.remaining() / (bytesPerSample * channels)
    val pcm = ByteArray(frameCount * 2)
    for (i in 0 until frameCount) {
      var acc = 0f
      for (c in 0 until channels) {
        acc += if (floatPcm) buf.float else buf.short / 32768f
      }
      val v = (acc / channels).coerceIn(-1f, 1f)
      val s = (v * 32767f).toInt()
      pcm[2 * i] = (s and 0xff).toByte()
      pcm[2 * i + 1] = ((s shr 8) and 0xff).toByte()
    }
    out.write(pcm)
    return frameCount.toLong()
  }

  private fun patchHeader(path: String, sampleRate: Int, frames: Long) {
    val data = (frames * 2).toInt()
    val h = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
    h.put("RIFF".toByteArray()).putInt(36 + data).put("WAVE".toByteArray())
    h.put("fmt ".toByteArray()).putInt(16).putShort(1).putShort(1).putInt(sampleRate)
      .putInt(sampleRate * 2).putShort(2).putShort(16)
    h.put("data".toByteArray()).putInt(data)
    RandomAccessFile(path, "rw").use { it.seek(0); it.write(h.array()) }
  }
}
