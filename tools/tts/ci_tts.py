#!/usr/bin/env python3
"""CI gate for on-device neural TTS: the app's TypeScript G2P + the C++ Kokoro engine,
checked for intelligibility with an independent speech recogniser.

For each voice and sentence: text -> phoneme tokens (tools/tts/phonemize.js, i.e. the
app's G2P) -> `mobigpt-rvc tts` (the engine that ships in the app) -> WAV ->
Moonshine ASR (sherpa-onnx) -> word error rate against the input text.

Usage:
  python tools/tts/ci_tts.py --cli build/rvc/mobigpt-rvc --model kokoro.int8.onnx \
      --voices voices-v1.0.bin --lexicon-dir LEX --asr-dir MOONSHINE --out build/tts-e2e
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

SENTENCES = [
    "Private artificial intelligence runs entirely on this phone.",
    "Nothing you type ever leaves the device.",
    "The quick brown fox jumps over the lazy dog.",
    "She sold six shiny shells on the shore, and her brother bought three of them.",
    "Please remind me to call my sister when I get home tonight.",
]
LONG = " ".join(
    [
        "Voice conversion used to need a powerful desktop computer.",
        "Today a phone can listen to your voice, understand its content, track the melody of your speech,",
        "and render it again in a completely different voice, one short chunk at a time.",
        "Every step runs locally, so recordings are never uploaded anywhere.",
    ]
    * 3
)
VOICES = ["af_heart", "am_michael", "bf_emma"]
MAX_WER = 0.15  # per sentence
MAX_MEAN_WER = 0.08


def words(text: str) -> list[str]:
    return re.sub(r"[^a-z' ]+", " ", text.lower().replace("-", " ")).split()


def wer(ref: str, hyp: str) -> float:
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / max(1, len(r))


def recogniser(asr_dir: str):
    import sherpa_onnx

    d = glob.glob(os.path.join(asr_dir, "sherpa-onnx-moonshine-*"))[0]
    return sherpa_onnx.OfflineRecognizer.from_moonshine(
        preprocessor=f"{d}/preprocess.onnx",
        encoder=f"{d}/encode.int8.onnx",
        uncached_decoder=f"{d}/uncached_decode.int8.onnx",
        cached_decoder=f"{d}/cached_decode.int8.onnx",
        tokens=f"{d}/tokens.txt",
        num_threads=4,
    )


def transcribe(rec, path: str) -> str:
    y, sr = sf.read(path, dtype="float32")
    s = rec.create_stream()
    s.accept_waveform(sr, y)
    rec.decode_stream(s)
    return s.result.text.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--voices", required=True, help="kokoro-onnx voices-v1.0.bin (npz)")
    ap.add_argument("--lexicon-dir", required=True)
    ap.add_argument("--asr-dir", required=True)
    ap.add_argument("--out", default="tts-e2e")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    npz = np.load(args.voices)
    rec = recogniser(args.asr_dir)

    def phonemize(text: str) -> dict:
        p = subprocess.run(["node", os.path.join(HERE, "phonemize.js"), "--lexicon-dir", args.lexicon_dir, text],
                           capture_output=True, text=True, check=True)
        return json.loads(p.stdout)

    def synth(tokens: dict, voice_bin: str, out: str, speed: float = 1.0) -> dict:
        p = subprocess.run([args.cli, "tts", "--model", args.model, "--voice", voice_bin, "--tokens", tokens["tokens"],
                            "--pauses", tokens["pauses"], "--speed", str(speed), out], capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(p.stderr.strip())
        return json.loads(p.stdout.strip().splitlines()[-1])

    rows, failures = [], []
    for voice in VOICES:
        vb = os.path.join(args.out, f"{voice}.bin")
        npz[voice].reshape(-1, 256).astype("<f4").tofile(vb)
        for i, text in enumerate(SENTENCES):
            out = os.path.join(args.out, f"{voice}-{i}.wav")
            ph = phonemize(text)
            st = synth(ph, vb, out)
            hyp = transcribe(rec, out)
            e = wer(text, hyp)
            rows.append({"voice": voice, "text": text, "heard": hyp, "wer": e, "rtf": st["rtf"], "seconds": st["seconds"],
                         "unknown": ph["unknown"]})
            if e > MAX_WER:
                failures.append(f"{voice} #{i}: WER {e:.2f} ({hyp!r})")

    # Long text: several model windows joined natively; check length and that speed scales duration.
    ph = phonemize(LONG)
    windows = ph["tokens"].count(";") + 1
    vb = os.path.join(args.out, "af_heart.bin")
    slow = synth(ph, vb, os.path.join(args.out, "long-1.0x.wav"), 1.0)
    fast = synth(ph, vb, os.path.join(args.out, "long-1.5x.wav"), 1.5)
    ratio = slow["seconds"] / fast["seconds"]
    if windows < 2:
        failures.append(f"long text produced {windows} window(s); windowing not exercised")
    if not 1.25 < ratio < 1.75:
        failures.append(f"speed 1.5x changed duration by {ratio:.2f}x")

    mean_wer = sum(r["wer"] for r in rows) / len(rows)
    if mean_wer > MAX_MEAN_WER:
        failures.append(f"mean WER {mean_wer:.3f} > {MAX_MEAN_WER}")

    lines = ["## Neural TTS (Kokoro-82M INT8) × Moonshine ASR", "",
             f"Mean WER **{mean_wer:.3f}** over {len(rows)} utterances · long text: {windows} windows, "
             f"{slow['seconds']:.1f} s at 1.0×, {fast['seconds']:.1f} s at 1.5× · RTF {slow['rtf']:.2f}", "",
             "| voice | WER | RTF | heard |", "|---|---|---|---|"]
    lines += [f"| {r['voice']} | {r['wer']:.2f} | {r['rtf']:.2f} | {r['heard']} |" for r in rows]
    if failures:
        lines += ["", "**Failures**", *[f"* {f}" for f in failures]]
    report = "\n".join(lines) + "\n"
    open(os.path.join(args.out, "report.md"), "w").write(report)
    json.dump({"rows": rows, "long": {"windows": windows, "slow": slow, "fast": fast}, "failures": failures},
              open(os.path.join(args.out, "results.json"), "w"), indent=2)
    print(report)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
