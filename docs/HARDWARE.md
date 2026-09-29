# Hardware requirements and warnings

Every optional feature declares requirements in `src/features/registry.ts`. They are checked against a live
`DeviceProfile` (`src/services/deviceProbe.ts`) by `src/features/requirements.ts`. Each check produces one of:

* **ok**: fully supported.
* **info**: works, with a note (e.g. missing `dotprod` means slower kernels).
* **warn**: works with caveats. The user sees the reasons and must confirm before enabling.
* **block**: cannot work on this device or build.

## Device tiers

| Tier | RAM | Recommended chat model (Q4) | Voice defaults |
|---|---|---|---|
| Entry | < 3 GB | ≤ 0.6 B (SmolLM2, Qwen3 0.6B) | DIO pitch, INT8 encoder, sequential loading |
| Low | 3–4 GB | ≤ 1.5 B | FCPE pitch, INT8, sequential (< 4 GB) |
| Mid | 6 GB | ≤ 3.2 B | RMVPE INT8, resident |
| High | 8 GB+ | ≤ 4.5 B | RMVPE INT8, resident, live mode OK on fast CPUs |
| Flagship | 12 GB+, 8 cores | ≤ 8.5 B | RMVPE + FP32 encoder |

Model fit uses llama.cpp's allocation model: weights, plus KV cache (layers × ctx × kv-heads × head-dim × type size, capped by sliding windows), plus compute buffers, with 10 % overhead. The result is compared against the RAM the OS realistically grants: about 60 % of total on Android and 55 % on iOS, or current free memory if that is higher.

## Feature requirements

| Feature | Blocks when | Warns when |
|---|---|---|
| Chat (core) | < 2 GB RAM, < 4 cores | < 4 GB RAM; no `dotprod` (info); emulator (info) |
| GPU offload | Android: GPU is not Adreno, or CPU lacks `dotprod`+`i8mm` (requirements of llama.rn's OpenCL build); iOS: no Metal GPU; any emulator/simulator | — |
| NPU (Hexagon) | iOS; no FastRPC (`libcdsprpc.so`) or HTP device; SoC older than Snapdragon 8 Gen 1 | always experimental; 8 Gen 1 (HTP v69) flagged as edge-of-support |
| Voice Studio | voice engine compiled out; < 2.5 GB RAM; < 4 cores; < 200 MB free storage | < 6 GB RAM; < 6 cores; no `dotprod`/`fp16` (info) |
| Live voice | voice engine compiled out; < 4 GB RAM; < 6 cores | < 8 GB RAM; < 8 cores; big core < 2.8 GHz; emulator |
| Text → voice | as Voice Studio | < 4 GB RAM |

Individual options carry their own requirements too. For example, RMVPE recommends 6 GB, the FP32 encoder recommends 8 GB, NNAPI/QNN are Android-only and Core ML is iOS-only.

## Approximate voice-engine memory while loaded

| Configuration | Files | Resident RAM* |
|---|---|---|
| INT8 ContentVec + DIO/PM/Harvest + voice | ≈ 95 MB + 55–110 MB | ≈ 350–450 MB |
| INT8 ContentVec + FCPE + voice | + 42 MB | ≈ 450–550 MB |
| INT8 ContentVec + RMVPE INT8 + voice | + 99 MB | ≈ 600–750 MB |
| FP32 ContentVec + RMVPE FP32 + voice | ≈ 740 MB + voice | ≈ 1.3–1.6 GB |
| Any of the above, sequential loading | — | ≈ largest single model + activations |

\*Includes ONNX Runtime arenas and activations for a 3.2 s window. Use **Benchmarks → Voice** to measure the real-time factor on your device.

## Architectures

* **Android:** `arm64-v8a` (all modern phones) and `x86_64` (emulators, Chromebooks, x86 tablets). 32-bit ABIs are not shipped because llama.cpp's Android build is 64-bit only, and Google Play requires 64-bit anyway.
* **iOS:** arm64 devices on iOS 15.1+, plus arm64/x86_64 simulators.
