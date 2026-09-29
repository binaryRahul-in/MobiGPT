// Android audio backend: android.media.AudioTrack / AudioRecord driven from
// native threads through JNI. PCM goes engine -> AudioTrack without ever
// touching JavaScript (or even Kotlin code).
#include <android/log.h>
#include <jni.h>

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "AudioIO.h"
#include "JniEnv.h"

#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "MobiGPTVoice", __VA_ARGS__)

namespace mobigpt::audio {

namespace {

constexpr jint STREAM_MUSIC = 3;
constexpr jint CHANNEL_OUT_MONO = 4;
constexpr jint CHANNEL_IN_MONO = 16;
constexpr jint ENCODING_PCM_FLOAT = 4;
constexpr jint MODE_STREAM = 1;
constexpr jint STATE_INITIALIZED = 1;
constexpr jint WRITE_BLOCKING = 0;
constexpr jint READ_BLOCKING = 0;
constexpr jint SOURCE_VOICE_RECOGNITION = 6;  // unprocessed-ish: no AGC / NS on most devices
constexpr jint SOURCE_MIC = 1;

bool clearException(JNIEnv* e, std::string* err, const char* what) {
  if (!e->ExceptionCheck()) return false;
  e->ExceptionClear();
  if (err) *err = what;
  return true;
}

class AudioTrackOutput final : public AudioOutput {
 public:
  ~AudioTrackOutput() override { stop(); }

  bool start(int sampleRate) override {
    JNIEnv* e = android::env();
    if (!e) return fail("no JNIEnv");
    jclass cls = e->FindClass("android/media/AudioTrack");
    if (!cls || clearException(e, &err_, "AudioTrack class missing")) return false;
    jmethodID minBuf = e->GetStaticMethodID(cls, "getMinBufferSize", "(III)I");
    const jint min = e->CallStaticIntMethod(cls, minBuf, sampleRate, CHANNEL_OUT_MONO, ENCODING_PCM_FLOAT);
    if (min <= 0) return fail("unsupported output rate " + std::to_string(sampleRate));
    const jint bytes = std::max(min * 2, sampleRate * 4 / 5);  // >= 200 ms of float audio
    jmethodID ctor = e->GetMethodID(cls, "<init>", "(IIIIII)V");
    jobject local = e->NewObject(cls, ctor, STREAM_MUSIC, sampleRate, CHANNEL_OUT_MONO, ENCODING_PCM_FLOAT, bytes, MODE_STREAM);
    if (!local || clearException(e, &err_, "AudioTrack() threw")) return false;
    track_ = e->NewGlobalRef(local);
    e->DeleteLocalRef(local);
    write_ = e->GetMethodID(cls, "write", "([FIII)I");
    play_ = e->GetMethodID(cls, "play", "()V");
    stop_ = e->GetMethodID(cls, "stop", "()V");
    pause_ = e->GetMethodID(cls, "pause", "()V");
    flush_ = e->GetMethodID(cls, "flush", "()V");
    release_ = e->GetMethodID(cls, "release", "()V");
    jmethodID state = e->GetMethodID(cls, "getState", "()I");
    if (e->CallIntMethod(track_, state) != STATE_INITIALIZED) {
      release(e);
      return fail("AudioTrack failed to initialise");
    }
    e->CallVoidMethod(track_, play_);
    if (clearException(e, &err_, "AudioTrack.play() threw")) {
      release(e);
      return false;
    }
    jfloatArray localArr = e->NewFloatArray(kBlock);
    jarray_ = static_cast<jfloatArray>(e->NewGlobalRef(localArr));
    e->DeleteLocalRef(localArr);
    e->DeleteLocalRef(cls);
    running_ = true;
    return true;
  }

  bool write(const float* s, size_t n) override {
    std::lock_guard<std::mutex> lk(m_);
    if (!running_ || !track_) return false;
    JNIEnv* e = android::env();
    if (!e) return false;
    while (n > 0 && running_) {
      const jint k = static_cast<jint>(std::min<size_t>(n, kBlock));
      e->SetFloatArrayRegion(jarray_, 0, k, s);
      const jint w = e->CallIntMethod(track_, write_, jarray_, 0, k, WRITE_BLOCKING);
      if (clearException(e, &err_, "AudioTrack.write() threw") || w < 0) {
        running_ = false;
        return false;
      }
      s += w;
      n -= static_cast<size_t>(w);
    }
    return running_;
  }

  void drain() override {
    std::lock_guard<std::mutex> lk(m_);
    JNIEnv* e = android::env();
    if (!e || !track_) return;
    // In MODE_STREAM, stop() lets queued audio play out before stopping.
    e->CallVoidMethod(track_, stop_);
    clearException(e, nullptr, "");
    release(e);
  }

  void stop() override {
    running_ = false;
    std::lock_guard<std::mutex> lk(m_);
    JNIEnv* e = android::env();
    if (!e || !track_) return;
    e->CallVoidMethod(track_, pause_);
    e->CallVoidMethod(track_, flush_);
    e->CallVoidMethod(track_, stop_);
    clearException(e, nullptr, "");
    release(e);
  }

  std::string error() const override { return err_; }

 private:
  static constexpr jint kBlock = 4096;
  bool fail(std::string msg) {
    err_ = std::move(msg);
    LOGE("%s", err_.c_str());
    return false;
  }
  void release(JNIEnv* e) {
    running_ = false;
    if (track_) {
      e->CallVoidMethod(track_, release_);
      clearException(e, nullptr, "");
      e->DeleteGlobalRef(track_);
      track_ = nullptr;
    }
    if (jarray_) {
      e->DeleteGlobalRef(jarray_);
      jarray_ = nullptr;
    }
  }

  std::mutex m_;
  std::atomic<bool> running_{false};
  jobject track_ = nullptr;
  jfloatArray jarray_ = nullptr;
  jmethodID write_{}, play_{}, stop_{}, pause_{}, flush_{}, release_{};
  std::string err_;
};

class AudioRecordInput final : public AudioInput {
 public:
  ~AudioRecordInput() override { stop(); }

  bool start(int preferred, Callback cb) override {
    JNIEnv* e = android::env();
    if (!e) {
      err_ = "no JNIEnv";
      return false;
    }
    jclass cls = e->FindClass("android/media/AudioRecord");
    if (!cls || clearException(e, &err_, "AudioRecord class missing")) return false;
    jmethodID minBuf = e->GetStaticMethodID(cls, "getMinBufferSize", "(III)I");
    jmethodID ctor = e->GetMethodID(cls, "<init>", "(IIIII)V");
    jmethodID state = e->GetMethodID(cls, "getState", "()I");
    jmethodID rel = e->GetMethodID(cls, "release", "()V");
    // 16 kHz avoids a resample; fall back to rates every device supports.
    for (int rate : {preferred, 16000, 48000, 44100}) {
      for (jint source : {SOURCE_VOICE_RECOGNITION, SOURCE_MIC}) {
        const jint min = e->CallStaticIntMethod(cls, minBuf, rate, CHANNEL_IN_MONO, ENCODING_PCM_FLOAT);
        if (min <= 0) continue;
        jobject local = e->NewObject(cls, ctor, source, rate, CHANNEL_IN_MONO, ENCODING_PCM_FLOAT, std::max(min * 2, rate * 4 / 5));
        if (!local || clearException(e, nullptr, "")) continue;
        if (e->CallIntMethod(local, state) == STATE_INITIALIZED) {
          rec_ = e->NewGlobalRef(local);
          e->DeleteLocalRef(local);
          rate_ = rate;
          break;
        }
        e->CallVoidMethod(local, rel);
        clearException(e, nullptr, "");
        e->DeleteLocalRef(local);
      }
      if (rec_) break;
    }
    if (!rec_) {
      err_ = "AudioRecord could not be opened (is RECORD_AUDIO granted?)";
      LOGE("%s", err_.c_str());
      return false;
    }
    read_ = e->GetMethodID(cls, "read", "([FIII)I");
    startRec_ = e->GetMethodID(cls, "startRecording", "()V");
    stopRec_ = e->GetMethodID(cls, "stop", "()V");
    release_ = rel;
    e->CallVoidMethod(rec_, startRec_);
    if (clearException(e, &err_, "startRecording() threw")) {
      releaseRecorder(e);
      return false;
    }
    running_ = true;
    thread_ = std::thread([this, cb = std::move(cb)] {
      JNIEnv* te = android::env();
      const jint block = rate_ / 50;  // 20 ms
      jfloatArray arr = te->NewFloatArray(block);
      std::vector<float> buf(static_cast<size_t>(block));
      while (running_) {
        const jint r = te->CallIntMethod(rec_, read_, arr, 0, block, READ_BLOCKING);
        if (te->ExceptionCheck()) {
          te->ExceptionClear();
          break;
        }
        if (r <= 0) continue;
        te->GetFloatArrayRegion(arr, 0, r, buf.data());
        cb(buf.data(), static_cast<size_t>(r));
      }
      te->DeleteLocalRef(arr);
    });
    e->DeleteLocalRef(cls);
    return true;
  }

  void stop() override {
    if (!rec_) return;
    running_ = false;
    JNIEnv* e = android::env();
    if (e) {
      e->CallVoidMethod(rec_, stopRec_);  // unblocks read()
      clearException(e, nullptr, "");
    }
    if (thread_.joinable()) thread_.join();
    if (e) releaseRecorder(e);
  }

  int sampleRate() const override { return rate_; }
  std::string error() const override { return err_; }

 private:
  void releaseRecorder(JNIEnv* e) {
    if (!rec_) return;
    e->CallVoidMethod(rec_, release_);
    clearException(e, nullptr, "");
    e->DeleteGlobalRef(rec_);
    rec_ = nullptr;
  }

  jobject rec_ = nullptr;
  jmethodID read_{}, startRec_{}, stopRec_{}, release_{};
  int rate_ = 16000;
  std::atomic<bool> running_{false};
  std::thread thread_;
  std::string err_;
};

}  // namespace

std::unique_ptr<AudioOutput> createAudioOutput() { return std::make_unique<AudioTrackOutput>(); }
std::unique_ptr<AudioInput> createAudioInput() { return std::make_unique<AudioRecordInput>(); }
void setHostOutputTap(std::function<void(const float*, size_t, int)>) {}

}  // namespace mobigpt::audio
