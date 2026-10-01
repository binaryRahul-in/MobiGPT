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
| **Accelerators** | Auto uses ONNX Runtime's CPU kernels (MLAS: NEON/dotprod on ARM64) on every platform. XNNPACK, NNAPI, QNN (Hexagon, needs the QNN build flavour) and Core ML are opt-in. XNNPACK was the Android default until the emulator e2e run caught it crashing inside ORT session creation for an RVC voice. | A provider that reports an error falls back to CPU, and the reason is shown in the UI. A crash inside ORT cannot fall back, which is why risky providers are opt-in. |

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

## INT8 quantisation analysis (measured)

Blind dynamic INT8 of ContentVec is known to hurt: voiceclonnx measured RVC's WER going from 38 % to 69 %.
MobiGPT's CI (`rvc-real-models` job) now **measures** every strategy against FP32 features on the same speech
(espeak-ng, 6 s) using `tools/rvc/quantize_rvc.py sweep`:

| ContentVec 768 variant | Size | Mean cosine | p5 cosine | Verdict |
|---|---|---|---|---|
| FP32 (reference) | 378 MB | 1.000 | 1.000 | — |
| **INT8 MatMul, per-channel, reduce_range (MobiGPT default)** | **122 MB** | **0.986** | **0.991** | ✅ shipped |
| INT8 with first/last 2 transformer layers in FP32 | 207 MB | 0.995 | 0.997 | ✅ highest fidelity |
| INT8 FFN-only (attention FP32) | 207 MB | 0.987 | 0.992 | ✅ |
| INT8 MatMul per-tensor, reduce_range | 122 MB | 0.974 | 0.965 | ✅ |
| upstream voiceclonnx `_q8` | 95 MB | 0.887 | 0.541 | ❌ fails gate |
| INT8 MatMul per-channel, full 8-bit range | 122 MB | 0.389 | 0.221 | ❌ saturates on x86 without VNNI |

Findings:
* **Per-channel weights plus 7-bit range** is the sweet spot: 3.1× smaller than FP32, with essentially identical features.
* **Full-range INT8 collapses on x86 CPUs without VNNI**, because U8S8 multiply-accumulates saturate. ARM phones don't saturate, but `reduce_range` costs nothing measurable, so it is used everywhere.
* **The upstream `_q8` misses the gate.** Its 5th-percentile cosine of 0.54 means some frames are badly distorted, which fits the WER regression voiceclonnx reported.

The chosen variant is built and parity-gated by `.github/workflows/publish-models.yml` and published as the
`models-v1` release asset `contentvec_768l12_int8_pc.onnx`. The app downloads it first and falls back to the upstream
file if the release is unreachable (for example offline mirrors or a blocked github.com).

| Component | Strategy | Metric | Gate |
|---|---|---|---|
| ContentVec / HuBERT | dynamic INT8, MatMul/Gemm, per-channel, reduce_range | frame-wise cosine vs FP32 | mean > 0.97, p5 > 0.90 |
| RMVPE / FCPE | dynamic INT8 MatMul/Gemm | F0 error in cents, voicing agreement | p95 < 50 cents, V/UV > 95 % |
| net_g (voice) | FP16 weight storage, FP32 compute (INT8 makes the vocoder sound metallic) | audible-band dB error vs FP32 | at run-to-run noise |

**Voices: FP16 weights, FP32 compute.** Measured on an RVC v2 40 kHz `net_g` exported with `export_voice_onnx.py`, on the same inputs:

| Voice variant | Size | x86 CPU time | Audible-band error vs FP32 |
|---|---|---|---|
| FP32 | 110 MB | 605 ms | 0.07 dB between consecutive FP32 runs (SineGen draws fresh noise) |
| **FP16 weights, FP32 compute** (`quantize --kind synth`, `export --fp16`) | **56 MB** | 680 ms | **0.1–0.4 dB** · waveform corr 0.9999+ |
| Full FP16 compute | 56 MB | 939 ms | 4.6 dB · corr 0.995 |

Weights are stored as FP16, each followed by a `Cast` to FP32 that ONNX Runtime folds away when the session is created. The download halves
and the change is at the level of the vocoder's own noise. CI (`tools/rvc/ci_export_voice.py`) exports a voice with the real RVC model code both ways
and gates the difference at 1 dB. In the real-model run, the 111 MB community voice becomes 56 MB and converts
speech with the same RTF and pitch tracking (`dio-fp16-voice` case). Full FP16 compute is both less accurate and slower on CPUs without native FP16 arithmetic, so it is only kept for
encoder experiments (`to_fp16`). That path repairs several `onnxconverter-common` defects that otherwise produce models ONNX Runtime refuses to load:
* graph-internal `Cast(to=FLOAT)` nodes whose outputs the converter retyped (HuBERT's attention mask);
* graph outputs that are also consumed inside the graph (w-okada's `unit12 → units9`);
* `RandomUniform`/`RandomNormal` (RVC's SineGen noise), which have no FP16 CPU kernel and so stay FP32 behind a `Cast`;
* `ConstantOfShape` fill values.

It also keeps the `pitchf → Sin/Cos/CumSum` phase path in FP32. `tools/rvc/test_quantize.py` runs every supported layout through FP16 and INT8 in CI.

### Where the time goes (x86 CI runner, 4 vCPU, 6 s of speech, 2.5 s chunks)

| Stage | RMVPE run | DIO run |
|---|---|---|
| ContentVec INT8 | 3.0 s | 3.1 s |
| Pitch | 3.5 s (RMVPE INT8) | 0.09 s (DIO) |
| net_g (community FP32 voice, 110 MB) | 10.9 s | 10.8 s |
| **Real-time factor** | 2.9× | 2.3× |

The synthesizer dominates. The DSP pitch trackers remove almost all of the pitch cost, which matters most on low-end phones.
Phones with 8 ARM cores are expected to run 2–4× faster than this shared runner. **Benchmarks → Voice**
measures the real figure on each device, and live mode shows a warning whenever RTF ≥ 1.

## Adding voices: .zip, .pth or .onnx, on the phone

Community RVC voices are usually shared as a `.zip` containing the `.pth` checkpoint and a FAISS `.index`
(for example `AIMan2001/PeterGriffin`). In **Voice → Library → Mine** you can pick that file, or paste its link
(on Hugging Face, the `…/resolve/main/…zip` download URL). The Hub search accepts `.zip`/`.pth` too. No PC is needed:

```
.zip ──► unzip (DEFLATE, zlib) ──► .pth = ZIP{data.pkl, data/<n>}
                                   │ a minimal pickle reader: torch.save's opcodes only, never executes code
                                   ▼
          version, sample rate, f0, speakers ──► rvc_template_<v>_<sr>.onnx (≈ 0.9 MB, downloaded once)
                                   │ every initializer = a checkpoint key, FP16, stored in an external file
                                   ▼
          weights.bin ◄── tensors copied as FP16 (FP32/BF16 converted, speaker table padded or truncated)
          model.onnx  ◄── the template, which ONNX Runtime loads with weights.bin next to it
```

The templates are RVC's own ONNX graph (`models_onnx.SynthesizerTrnMsNSFsidM`, commit 7ef1986), exported once per
architecture (v1/v2 × 32/40/48 kHz) by `tools/rvc/make_voice_templates.py` from unique random values, so that every
initializer maps 1:1 to a checkpoint key (457 for v2). An imported voice matches `export_voice_onnx.py --fp16` (correlation
0.9999, checked in CI). The `.index` is ignored because MobiGPT runs with `index_rate = 0`. Voices trained without pitch
guidance (f0 = 0) are not supported yet.

`.github/workflows/inspect-voice.yml` answers "will this voice work?" for any link: it reports the archive's contents,
converts it both ways (PC exporter and on-device importer), and runs speech through the engine. For `AIMan2001/PeterGriffin`
it found RVC v2, 48 kHz, pitch-guided, 375 epochs, 457 FP16 tensors, and the output pitch followed the input (1.00) with DIO
and FCPE.

The PC route still works and produces a single `.onnx`:

```bash
git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI rvc
git -C rvc checkout 7ef1986   # RVC 2.3 removed the ONNX model definitions; this is the last commit with them (and with the export fix)
pip install torch onnx onnxconverter-common
python tools/rvc/export_voice_onnx.py --rvc-repo rvc my_voice.pth my_voice.onnx --fp16   # --fp16: half-size weights, same quality
```

## Neural text-to-speech (Kokoro-82M)

Voice Studio's **Text** mode can speak with [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0), a
StyleTTS2/ISTFTNet model, running in the same C++ engine as RVC. The output can be played as is or converted into the
selected RVC voice. The operating system's voice is still available as an alternative.

```
text ──► TypeScript G2P (src/services/tts) ──► phoneme-token windows (≤ 510 ids)
          misaki gold+silver lexicons,           │   (only small int arrays cross JSI)
          -s/-ed/-ing rules, numbers, spelling   ▼
                                    C++ KokoroModel (cpp/tts) on ONNX Runtime
                                    style row = voice[token count − 1], speed
                                                 ▼
                                    24 kHz WAV ──► play, or ──► RVC conversion
```

| Piece | Details |
|---|---|
| Model | `kokoro-v1.0.int8.onnx` (92 MB). Both export layouts are detected: kokoro-onnx (`tokens → audio`) and onnx-community (`input_ids → waveform`). `speed` may be float or int. |
| Voices | Seven English styles (`af_heart`, `af_bella`, `af_sarah`, `am_michael`, `am_fenrir`, `bf_emma`, `bm_george`), 522 KB each, as raw `510 × 256` float32 |
| G2P | A port of misaki's lexicon path: gold/silver US dictionaries (≈ 180 k words), DEFAULT reading for heteronyms, misaki's `-s/-ed/-ing` morphology, numbers/years/ordinals/decimals read as words, compounds split, acronyms and unknown words spelled. misaki's spaCy part-of-speech tagging is not ported. |
| Windows | Text is split at sentence ends, then clauses, then words, so no window exceeds 510 tokens; windows are joined natively with short pauses |
| Sources | The `models-v1` release mirror first, then the upstream GitHub release / misaki repo, then Hugging Face (`onnx-community/Kokoro-82M-v1.0-ONNX`) |

**Measured** (`tools/tts/ci_tts.py`, CI job `tts-real-model`): 15 sentences across three voices were synthesised through the
app's G2P and the C++ engine, then transcribed by an independent recogniser (Moonshine tiny, sherpa-onnx). Word error rate
was **0.000**. Speed is RTF ≈ 0.75–0.8 on a 4-vCPU x86 runner with the INT8 model. A 12-sentence paragraph is split into
3 windows, and 1.5× speed shortens it by 1.54×.
