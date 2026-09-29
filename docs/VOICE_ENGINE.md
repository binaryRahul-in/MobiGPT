# MobiGPT voice engine (RVC on ONNX Runtime)

MobiGPT's v2 feature converts speech into another person's voice using
[Retrieval-based Voice Conversion](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI).
The whole pipeline runs on the device, in C++, and was designed around the limits of phones.

## Pipeline

```
input (mic · file · offline TTS)
   │  any rate / format ──► native decode (MediaCodec / AVAudioFile) ──► Kaiser-sinc resampler ──► 16 kHz mono
   ▼
StreamingConverter (2–3 s core windows, ±0.3 s context, 60 ms SOLA crossfade)
   │ per window
   ├─► ContentEncoder   HuBERT / ContentVec  → 768-d (v2) or 256-d (v1) features @ 50 Hz → nearest 2× → 100 Hz
   ├─► PitchExtractor   RMVPE | FCPE | Harvest | DIO+StoneMask | PM   → F0 @ 100 Hz → ±semitones → coarse (1..255)
   └─► Synthesizer      net_g (per voice) → 40 / 48 / 32 kHz audio  → rms_mix_rate envelope transfer
   ▼
SOLA stitch (sample-exact length) ──► AudioTrack (Android) / AVAudioPlayerNode (iOS) or WAV file
```

Source: `packages/react-native-mobigpt-voice/cpp/rvc/`

| File | Responsibility |
|---|---|
| `Audio.*` | WAV I/O (8/16/24/32-bit PCM, float; streaming writer); band-limited streaming resampler |
| `Dsp.*` | radix-2 FFT (fp32/fp64), Slaney/HTK mel filterbanks, RMVPE and FCPE log-mel front-ends |
| `Pitch.*` | RMVPE and FCPE (ONNX, two I/O layouts each), WORLD Harvest/DIO(+StoneMask), PM tracker, reference decoders |
| `OrtSession.*` | ONNX Runtime sessions, accelerator selection with automatic CPU fallback, I/O introspection, fp16 helpers |
| `Models.*` | content encoder and synthesizer adapters for RVC-WebUI, transformers and w-okada exports |
| `Pipeline.*` | `RvcEngine` (resident/sequential loading), `StreamingConverter` (chunking + SOLA), `mixRms` |
| `VoiceService.*` | engine lifetime, live mic→speaker session, recording, playback, benchmark, model inspector |
| `jsi/VoiceJSI.*` | `global.__MobiGPTVoice`: Promise-based API running on worker threads |

## The mobile optimisations

| Optimisation | How it works here | Effect |
|---|---|---|
| **Pitch extractor swap** | `PitchMethod` is selected per load. Choices: RMVPE (ONNX U-Net), FCPE (ONNX, ~42 MB), or WORLD Harvest, DIO+StoneMask and PM (normalised autocorrelation with window correction), which are native C++ with no model file. | DSP trackers need **0 MB** of extra RAM and no download. FCPE is roughly 5–10× lighter than RMVPE. |
| **Drop the FAISS index** | `PipelineConfig.indexRate` is forced to `0` in `RvcEngine`'s constructor, and a warning is surfaced if a caller passes anything else. `.index` files are never read. | Saves **100–500 MB** per voice and removes a slow load step. |
| **Chunked streaming** | `StreamingConverter` processes `2 × context + core + crossfade` windows. HuBERT alignment is exact (320-sample multiples) and SOLA picks the best overlap offset within ±10 ms. | Memory is flat for any input length. Output length is exactly `ceil(N/160) × hop`. A test verifies an identity model is reproduced bit-for-bit with random block sizes. |
| **Tensors stay in C++** | JS only sends paths and configuration. HuBERT, pitch and net_g tensors never leave C++, and PCM goes directly to `AudioTrack` (via JNI from the worker thread) or `AVAudioPlayerNode`. | No JS↔native serialisation and no GC pressure. |
| **Sequential loading** | `LoadStrategy::Sequential` runs every window through the encoder first (features stored as fp16), then pitch, then the synthesizer. Only one model is resident at a time. | Peak RAM ≈ the largest single model. This is the default on phones with less than 4 GB. |
| **Low-memory sessions** | `SessionConfig.lowMemory` disables ORT's CPU arena and memory patterns. | Lower resident memory between runs, at 5–15 % lower speed. |
| **Accelerators** | Auto uses XNNPACK on Android and MLAS on iOS. Optional: NNAPI, QNN (Hexagon, needs the QNN build flavour) and Core ML. | Any provider that fails falls back to CPU, and the reason is shown in the UI. |

## Supported model formats

| Component | Layouts auto-detected |
|---|---|
| Content encoder | `source[1,1,N] → embed` (RVC WebUI), `input_values[1,N] (+attention_mask) → hidden_states` (transformers/voiceclonnx), `audio[1,N] → unit12/units9/unit12s` (w-okada, output chosen from voice metadata) |
| RMVPE | `input[1,128,T] → [1,T,360]` salience (decoded in C++), `waveform[1,N] + threshold[1] → pitchf` (w-okada) |
| FCPE | `mel[1,T,128] → latent[1,T,360]`, `[1,128,T]`, or waveform-input variants that return Hz directly |
| Voice (net_g) | RVC WebUI `phone, phone_lengths, pitch, pitchf, ds, rnd → audio`; w-okada `feats, p_len, pitch, pitchf, sid → audio`; fp16 I/O; no-f0 voices; v1 (256-d) and v2 (768-d) |

The sample rate comes from the `sample_rate` metadata key or w-okada's JSON `metadata.samplingRate`. If neither is present it is **measured** with a 20-frame warm-up inference (hop × 100). Imported files are inspected natively. A file that isn't a synthesizer is rejected with an explanation (for example "this is an encoder model").

## Numerical verification

The C++ tests (`tests/`, 29 cases, run on Linux, macOS and under ASan+UBSan in CI) check the following:

* **Mel front-ends vs librosa:** filterbanks match to within 1e-6, and RMVPE/FCPE log-mel spectrograms to max |Δ| < 5e-4.
* **RMVPE / FCPE decoders:** match the reference formulas exactly.
* **DSP trackers:** DIO, Harvest and PM recover 110/220/330 Hz within 3 %.
* **Resampler:** frequency and loudness are preserved, and there is no aliasing at 9 kHz → 16 kHz.
* **Every model layout:** loads and produces exactly `frames × hop` samples.
* **Pitch shift:** measured end-to-end at 0, +12 and −5 semitones (±3 %) with a phase-continuous vocoder fixture.
* **Chunking and sequential loading:** no clicks across chunk seams at 2, 2.5 and 3 s chunks. Sequential output equals resident output within fp16 tolerance.
* **Live mode, cancellation and inspection:** live mode runs natively with a null audio backend; cancellation works; the inspector classifies every file type.

The `rvc-real-models` CI job then downloads real community models and runs espeak-ng speech through the engine CLI with every pitch method. It checks duration, loudness and that the output F0 follows the input (including +12 semitones), and publishes the audio as an artifact.

## INT8 quantisation analysis

Blind dynamic INT8 of ContentVec is harmful. voiceclonnx measured 38 % → 69 % WER on RVC. The damage comes from the convolutional feature extractor and the positional convolution, whose activations have large outliers.

MobiGPT's `tools/rvc/quantize_rvc.py` quantises selectively and measures the result:

| Component | Strategy | Metric reported | Pass threshold |
|---|---|---|---|
| ContentVec / HuBERT | dynamic INT8, **MatMul/Gemm only**, per-channel (transformer ≈ 85 % of weights; conv front-end kept fp32) | frame-wise cosine similarity vs FP32 | mean > 0.97, p5 > 0.90 |
| RMVPE | dynamic INT8 MatMul/Gemm (optionally GRU) | F0 error in cents, voicing agreement | p95 < 50 cents, V/UV > 95 % |
| FCPE | dynamic INT8 MatMul/Gemm | same as RMVPE | same |
| net_g (voice) | **FP16 weights, fp32 I/O** (INT8 makes the HiFi-GAN-style decoder sound metallic) | log-mel RMSE | < 0.5 |

```bash
python tools/rvc/quantize_rvc.py analyze contentvec_768l12.onnx            # op histogram + weight share per op
python tools/rvc/quantize_rvc.py quantize --kind encoder --audio me.wav contentvec.onnx contentvec_q8.onnx --report r.json
python tools/rvc/quantize_rvc.py quantize --kind synth voice.onnx voice_fp16.onnx
```

In CI, the real-model job quantises the FP32 ContentVec with this selective strategy and compares it with the upstream `_q8` file. Both are scored by cosine similarity against FP32 features on the same speech, and the table is written to the job summary.

## Converting your own voices

```bash
git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI rvc
python tools/rvc/export_voice_onnx.py --rvc-repo rvc my_voice.pth my_voice.onnx --fp16
```

Import the `.onnx` in **Voice → Library → Mine → Import**, or push it to a Hugging Face repo and use **Hub** search.
