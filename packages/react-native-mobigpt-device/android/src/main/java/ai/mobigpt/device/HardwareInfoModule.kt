package ai.mobigpt.device

// Adapted from PocketPal AI (MIT, (c) Asghar Ghorbani) — see NOTICE.md.
import android.app.ActivityManager
import android.content.Context
import android.opengl.GLES20
import android.os.Build
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.module.annotations.ReactModule
import java.io.File
import javax.microedition.khronos.egl.EGL10
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.egl.EGLContext

@ReactModule(name = NativeHardwareInfoSpec.NAME)
class HardwareInfoModule(context: ReactApplicationContext) : NativeHardwareInfoSpec(context) {

  override fun getName(): String = NativeHardwareInfoSpec.NAME

  override fun getCPUInfo(promise: Promise) {
    try {
      val features = mutableSetOf<String>()
      var hardware = Build.HARDWARE ?: ""
      val cpuinfo = File("/proc/cpuinfo")
      if (cpuinfo.canRead()) {
        cpuinfo.readLines().forEach { line ->
          val parts = line.split(":", limit = 2)
          if (parts.size == 2) {
            when (parts[0].trim()) {
              "Features", "flags" -> features.addAll(parts[1].trim().split(" ").filter { it.isNotEmpty() })
              "Hardware" -> if (parts[1].isNotBlank()) hardware = parts[1].trim()
            }
          }
        }
      }
      val map = Arguments.createMap()
      map.putInt("cores", Runtime.getRuntime().availableProcessors())
      val arr = Arguments.createArray()
      features.sorted().forEach { arr.pushString(it) }
      map.putArray("features", arr)
      map.putBoolean("hasFp16", features.any { it == "fphp" || it == "asimdhp" || it == "f16c" })
      map.putBoolean("hasDotProd", features.any { it == "asimddp" || it == "dotprod" || it == "avx_vnni" })
      map.putBoolean("hasSve", features.contains("sve"))
      map.putBoolean("hasI8mm", features.contains("i8mm"))
      map.putString("socModel", if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) Build.SOC_MODEL ?: "" else "")
      map.putString("hardware", hardware)
      map.putInt("maxFreqMhz", maxFreqMhz())
      val abis = Arguments.createArray()
      Build.SUPPORTED_ABIS.forEach { abis.pushString(it) }
      map.putArray("abis", abis)
      promise.resolve(map)
    } catch (e: Exception) {
      promise.reject("E_CPU", e.message, e)
    }
  }

  override fun getGPUInfo(promise: Promise) {
    var renderer = ""
    var vendor = ""
    var version = ""
    try {
      val egl = EGLContext.getEGL() as EGL10
      val display = egl.eglGetDisplay(EGL10.EGL_DEFAULT_DISPLAY)
      if (display != EGL10.EGL_NO_DISPLAY) {
        egl.eglInitialize(display, IntArray(2))
        val configs = arrayOfNulls<EGLConfig>(1)
        val count = IntArray(1)
        egl.eglChooseConfig(display, intArrayOf(EGL10.EGL_RENDERABLE_TYPE, 4, EGL10.EGL_NONE), configs, 1, count)
        if (count[0] > 0) {
          val ctx = egl.eglCreateContext(display, configs[0], EGL10.EGL_NO_CONTEXT, intArrayOf(0x3098, 2, EGL10.EGL_NONE))
          val surface = egl.eglCreatePbufferSurface(display, configs[0], intArrayOf(EGL10.EGL_WIDTH, 1, EGL10.EGL_HEIGHT, 1, EGL10.EGL_NONE))
          if (ctx != EGL10.EGL_NO_CONTEXT && surface != EGL10.EGL_NO_SURFACE) {
            egl.eglMakeCurrent(display, surface, surface, ctx)
            renderer = GLES20.glGetString(GLES20.GL_RENDERER) ?: ""
            vendor = GLES20.glGetString(GLES20.GL_VENDOR) ?: ""
            version = GLES20.glGetString(GLES20.GL_VERSION) ?: ""
            egl.eglMakeCurrent(display, EGL10.EGL_NO_SURFACE, EGL10.EGL_NO_SURFACE, EGL10.EGL_NO_CONTEXT)
            egl.eglDestroySurface(display, surface)
            egl.eglDestroyContext(display, ctx)
          }
        }
        egl.eglTerminate(display)
      }
    } catch (_: Throwable) {
      // GPU details are best effort (e.g. headless emulators).
    }
    val r = renderer.lowercase()
    val adreno = Regex("adreno|qualcomm").containsMatchIn(r)
    val mali = r.contains("mali")
    val powervr = r.contains("powervr")
    val map = Arguments.createMap()
    map.putString("renderer", renderer)
    map.putString("vendor", vendor)
    map.putString("version", version)
    map.putBoolean("hasAdreno", adreno)
    map.putBoolean("hasMali", mali)
    map.putBoolean("hasPowerVR", powervr)
    map.putBoolean("hasAppleGpu", false)
    map.putString(
      "gpuType",
      when {
        adreno -> "Adreno (Qualcomm)"
        mali -> "Mali (Arm)"
        powervr -> "PowerVR"
        renderer.isNotEmpty() -> renderer
        else -> "Unknown"
      },
    )
    promise.resolve(map)
  }

  override fun getAvailableMemory(promise: Promise) {
    val am = reactApplicationContext.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    val info = ActivityManager.MemoryInfo()
    am.getMemoryInfo(info)
    promise.resolve(info.availMem.toDouble())
  }

  override fun hasNpu(promise: Promise) {
    // Hexagon DSP/HTP is reachable through the FastRPC library.
    val candidates = listOf("/vendor/lib64/libcdsprpc.so", "/system/vendor/lib64/libcdsprpc.so", "/vendor/lib/libcdsprpc.so")
    promise.resolve(candidates.any { File(it).exists() })
  }

  private fun maxFreqMhz(): Int {
    var maxKhz = 0L
    for (core in 0 until Runtime.getRuntime().availableProcessors().coerceIn(1, 16)) {
      try {
        val f = File("/sys/devices/system/cpu/cpu$core/cpufreq/cpuinfo_max_freq")
        if (f.canRead()) maxKhz = maxOf(maxKhz, f.readText().trim().toLongOrNull() ?: 0L)
      } catch (_: Throwable) {
      }
    }
    return (maxKhz / 1000).toInt()
  }
}
