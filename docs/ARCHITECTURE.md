# Architecture

```
┌──────────────────────────── React Native (TypeScript) ────────────────────────────┐
│ screens/  Onboarding · Chat · Models · HF browser · Voice Studio · Voice library ·│
│           Device · Benchmarks · Features · Settings · About · History              │
│ stores/   Settings · Device · Models · Chat · Voice · Benchmark · Update (MobX)    │
│ features/ DeviceProfile → requirement engine → feature registry (options, packs)   │
│ services/ llm (llama.rn façade) · hf · downloads · deviceProbe · updates · paths   │
└───────────────┬──────────────────────────┬───────────────────────────┬─────────────┘
                │ JSI                       │ JSI (global.__MobiGPTVoice) │ TurboModule
        ┌───────▼───────┐        ┌──────────▼──────────────┐     ┌──────▼──────────────┐
        │ llama.rn      │        │ react-native-mobigpt-   │     │ react-native-       │
        │ llama.cpp     │        │ voice (C++ RVC engine)  │     │ mobigpt-device      │
        │ CPU/OpenCL/   │        │ ONNX Runtime · WORLD ·  │     │ CPU/GPU/NPU/RAM     │
        │ Metal/Hexagon │        │ AudioTrack/AVAudioEngine│     │ introspection       │
        └───────────────┘        └─────────────────────────┘     └─────────────────────┘
```

## Principles

1. **Local first.** The network is used only for model downloads (Hugging Face), the remote catalogue (`catalog/*.json` on the `main` branch) and GitHub Releases update checks.
2. **Measure before recommending.** Each screen reads from one `DeviceProfile`. Model cards show an estimated RAM need and a fit badge, and features show requirement evaluations.
3. **Everything heavy is optional.** Features are opt-in with confirmation dialogs for warnings. Runtime packs (models, encoders, pitch trackers, voices) are downloaded on demand, and the voice native module can be removed at build time.
4. **Native code is testable on a laptop.** The RVC engine is plain C++17/20 with a platform audio interface, so it builds and runs its full test suite on Linux and macOS in CI.

## State and persistence

MobX stores are persisted with `mobx-persist-store` to AsyncStorage: settings, downloaded models, conversations, installed voice packs and benchmark history. Model files live under `Documents/mobigpt/…`. Downloads go to a `.part` file that is size-checked and then atomically renamed.

## Build-time configuration

`mobigpt.features.json` is read by:
* `react-native.config.js`: disables autolinking of the voice package when `voice.enabled` is false;
* `android/build.gradle`: sets the ONNX Runtime version and flavour (`standard` or `qnn`) for the voice module;
* `scripts/test-native.sh` and CI: choose the host ONNX Runtime version.

## Testing pyramid

| Layer | Tooling | Where |
|---|---|---|
| C++ engine | custom harness + CMake, ASan/UBSan | `packages/react-native-mobigpt-voice/tests` |
| Real models | engine CLI + librosa pYIN checks | `tools/rvc/ci_real_models.py` |
| Logic and stores | Jest with in-memory FS, fake LLM and fake voice engine | `__tests__/logic.test.ts`, `__tests__/stores.test.ts` |
| UI integration | Testing Library rendering the real app | `__tests__/App.test.tsx` |
| Device end-to-end | Maestro on an Android emulator | `e2e/maestro` |
| Builds | Gradle release APK (native-lib verification), Xcode simulator build | `.github/workflows` |
