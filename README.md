<p align="center">
  <img src="assets/brand/mobigpt-icon-512.png" width="112" alt="MobiGPT" />
</p>

<h1 align="center">MobiGPT</h1>

<p align="center"><b>Private AI that runs entirely on your phone.</b><br/>
Chat with open LLMs through llama.cpp and turn any speech into another voice with on-device RVC.<br/>
Android (arm64 · x86_64) and iOS. No account, no cloud, no telemetry.</p>

<p align="center">
  <a href="../../actions/workflows/ci.yml"><img alt="CI" src="../../actions/workflows/ci.yml/badge.svg"></a>
  <a href="../../actions/workflows/android.yml"><img alt="Android" src="../../actions/workflows/android.yml/badge.svg"></a>
  <a href="../../actions/workflows/ios.yml"><img alt="iOS" src="../../actions/workflows/ios.yml/badge.svg"></a>
</p>

---

## Features

| | |
|---|---|
| **On-device chat** | GGUF models (Llama 3.2, Qwen3, Gemma 3, Phi-3.5, SmolLM2…) via [llama.rn](https://github.com/mybigday/llama.rn). Streaming, stop, thinking mode, conversation history. |
| **Hardware-aware** | Detects RAM, CPU features (dotprod, i8mm, fp16), GPU (Adreno / Mali / Apple) and NPU (Hexagon HTP). Shows which models fit before you download them. |
| **CPU · GPU · NPU** | CPU everywhere, OpenCL on Adreno, Metal on Apple silicon, and experimental Hexagon NPU offload. |
| **Model hub** | Curated catalogue (updated remotely), Hugging Face search with per-file fit badges, import `.gguf` from storage, download/pause/delete, load/unload. |
| **Benchmarks** | llama-bench style prompt/generation throughput, plus the voice engine's real-time factor. Results are kept as history. |
| **Voice Studio (v2)** | Retrieval-based Voice Conversion on ONNX Runtime. Record, convert files (mp3/m4a/wav…), type text (offline TTS → RVC), or run a live mic→speaker voice changer. |
| **Optional & modular** | Every heavy feature is opt-in, with hardware warnings. The voice engine can also be compiled out at build time. |
| **Updates** | In-app check against GitHub Releases, plus a remote model/voice catalogue. |

## The voice engine at a glance

```
mic / file / TTS ─► 16 kHz ─┬─► ContentVec / HuBERT (INT8) ──► 2× upsample ─┐
                            └─► pitch: RMVPE · FCPE · Harvest · DIO · PM ─────┴─► net_g (RVC voice) ─► 40/48 kHz ─► AudioTrack / AVAudioEngine
                                 ▲ chunked streaming (2–3 s windows + context, SOLA crossfade) — flat memory
                                 ▲ index_rate = 0 — FAISS index never loaded
                                 ▲ all tensors stay in C++ (JSI) — nothing crosses into JavaScript
```

* **Pitch extractor swap:** RMVPE (neural, best), FCPE (lightweight ONNX), or zero-RAM native DSP (WORLD Harvest, DIO + StoneMask, and a PM autocorrelation tracker).
* **No FAISS:** `index_rate` is hard-wired to 0, which saves 100–500 MB per voice.
* **Chunked streaming:** fixed windows with context and SOLA stitching. Output is sample-exact and memory stays flat.
* **Native tensor path:** C++ (ONNX Runtime C++ API) end to end, bound to JS with JSI. Output PCM goes straight to Android `AudioTrack` or iOS `AVAudioEngine`.
* **Accepts both export formats:** RVC WebUI (`phone/phone_lengths/pitch/pitchf/ds/rnd`) and w-okada (`feats/p_len/sid` + metadata). Handles v1/v2, fp16 I/O and no-f0 voices, and validates imported models before adding them.
* **INT8 done carefully:** selective MatMul-only quantisation with a measured parity report (`tools/rvc/quantize_rvc.py`). See [docs/VOICE_ENGINE.md](docs/VOICE_ENGINE.md).

## Repository layout

```
src/                         React Native app (TypeScript, MobX, React Native Paper)
  features/                  device profile, requirement engine, feature registry
  services/                  llama.rn façade, Hugging Face client, downloads, updates
  stores/                    app state (models, chat, voice, benchmarks, settings)
  screens/                   Onboarding, Chat, Models, HF browser, Voice Studio, Voice library,
                             Device, Benchmarks, Features, Settings, About
packages/
  react-native-mobigpt-voice/   RVC engine: C++ core + JSI + Android (JNI/AudioTrack) + iOS (AVAudioEngine)
    cpp/rvc/                    pipeline, pitch trackers, mel front-end, ORT sessions, SOLA stitcher
    cpp/third_party/world/      WORLD (BSD) — DIO / Harvest / StoneMask
    tests/                      29 host tests (librosa-verified DSP, all model layouts, live mode…)
  react-native-mobigpt-device/  CPU/GPU/NPU/memory introspection TurboModule
catalog/                     remote-updatable model and voice catalogues
tools/rvc/                   INT8/FP16 quantiser with parity checks, .pth → ONNX exporter, CI real-model test
e2e/maestro/                 end-to-end flows (onboarding, navigation, voice setup, download + chat)
docs/                        architecture, voice engine, hardware requirements
```

## Build

Requirements: Node 22, Yarn 1, JDK 17, Android SDK + NDK 27, Xcode 16+ with CocoaPods.

```bash
yarn install                 # also fetches llama.rn native artifacts
yarn android                 # or: cd android && ./gradlew assembleRelease
cd ios && bundle exec pod install && cd .. && yarn ios
```

### Build-time feature switches

`mobigpt.features.json` controls what is compiled in:

```jsonc
{ "voice": { "enabled": true, "ortVersion": "1.24.3", "ortFlavor": "standard" } }
```

* `voice.enabled: false` (or `MOBIGPT_VOICE=0`) leaves the ONNX Runtime/RVC native module out of the build (≈20 MB less per ABI). The app hides Voice Studio instead of crashing.
* `voice.ortFlavor: "qnn"` links ONNX Runtime's QNN build, which enables the Hexagon NPU for the voice engine.

## Tests

```bash
yarn typecheck && yarn lint && yarn test     # 37 unit + UI integration tests (Jest, Testing Library)
yarn test:native                             # C++ engine: 29 tests, downloads ONNX Runtime; SANITIZE=1 for ASan/UBSan
maestro test e2e/maestro                     # end-to-end on a device/emulator
```

CI runs on every push:
- **CI:** JS checks and tests; the C++ engine on Linux and macOS, including under ASan/UBSan; real community RVC models run through the engine CLI; catalogue validation against Hugging Face.
- **Android:** release APK build, native-library verification, and Maestro end-to-end flows on an emulator (including downloading a model and chatting with it).
- **iOS:** Release simulator build.

## Documentation

* [Architecture](docs/ARCHITECTURE.md)
* [Voice engine (RVC) and INT8 analysis](docs/VOICE_ENGINE.md)
* [Hardware requirements and warnings](docs/HARDWARE.md)
* [Third-party notices](NOTICE.md)

MobiGPT was inspired by [PocketPal AI](https://github.com/a-ghorbani/pocketpal-ai) (MIT). It is a separate app with its own branding, and a few helpers are adapted from PocketPal with attribution (see NOTICE.md).

## License

MIT. See [LICENSE](LICENSE).
