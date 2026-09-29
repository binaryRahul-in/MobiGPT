#!/usr/bin/env python3
"""
End-to-end check of the C++ RVC engine against *real* community models.

  1. downloads ContentVec (INT8 + FP32), RMVPE (INT8) and a community voice
  2. synthesises an English sentence with espeak-ng
  3. converts it with every pitch extractor through the host CLI (the exact
     C++ engine that ships inside the app)
  4. asserts: output length == input length, audio is not silent/NaN, output
     F0 tracks input F0 (and doubles with +12 semitones)
  5. INT8 analysis: selective INT8 vs upstream INT8 vs FP32 ContentVec features

Writes a Markdown report (for $GITHUB_STEP_SUMMARY) and JSON results.
Usage: python ci_real_models.py --cli build/mobigpt-rvc --out out/
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

BASE_REPO = "TigreGotico/voiceclonnx-rvc"
VOICE_REPO = "ozada/onnx_rvc"


def hf_file(repo: str, name: str, out: str) -> str:
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(repo)
    match = next((f for f in files if f == name), None) or next((f for f in files if f.endswith("/" + name)), None)
    if not match:
        raise FileNotFoundError(f"{name} not found in {repo}: {files[:20]}")
    return hf_hub_download(repo, match, local_dir=out)


def first_voice(repo: str, out: str) -> str:
    from huggingface_hub import hf_hub_download, list_repo_files

    onnx = [f for f in list_repo_files(repo) if f.endswith(".onnx") and not any(k in f.lower() for k in ("hubert", "vec", "rmvpe"))]
    pick = next((f for f in onnx if "woman_1" in f), onnx[0])
    return hf_hub_download(repo, pick, local_dir=out)


def f0_median(path: str) -> float:
    import librosa
    import soundfile as sf

    y, sr = sf.read(path, dtype="float32", always_2d=False)
    y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=900, sr=16000, frame_length=1024)
    v = f0[voiced & np.isfinite(f0)]
    return float(np.median(v)) if len(v) else 0.0


def run(cli: str, args: list[str]) -> dict:
    p = subprocess.run([cli, *args], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or p.stdout.strip())
    return json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip().startswith("{") else {"stdout": p.stdout, "stderr": p.stderr}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--out", default="rvc-e2e")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    models = os.path.join(args.out, "models")

    enc_q8 = hf_file(BASE_REPO, "contentvec_768l12_q8.onnx", models)
    rmvpe = hf_file(BASE_REPO, "rmvpe_q8.onnx", models)
    # The FCPE pack phones with 4-6 GB RAM get by default (catalog: niobures/FCPE).
    fcpe = hf_file("niobures/FCPE", "onnx/fcpe.onnx", models)
    voice = first_voice(VOICE_REPO, models)

    speech = os.path.join(args.out, "speech.wav")
    subprocess.run(["espeak-ng", "-v", "en-us", "-s", "150", "-w", speech,
                    "Private artificial intelligence runs entirely on this phone. Nothing leaves the device."], check=True)
    import soundfile as sf

    x, sr = sf.read(speech)
    in_dur = len(x) / sr
    in_f0 = f0_median(speech)

    info = {m: run(args.cli, ["inspect", m]) for m in (enc_q8, rmvpe, fcpe, voice)}
    results = []
    cases = [
        ("rmvpe", ["--pitch", "rmvpe", "--pitch-model", rmvpe]),
        ("dio", ["--pitch", "dio"]),
        ("harvest", ["--pitch", "harvest"]),
        ("pm", ["--pitch", "pm"]),
        ("rmvpe+12", ["--pitch", "rmvpe", "--pitch-model", rmvpe, "--key", "12"]),
        ("dio-sequential", ["--pitch", "dio", "--sequential"]),
        ("dio-chunk2", ["--pitch", "dio", "--chunk", "2.0"]),
        ("fcpe", ["--pitch", "fcpe", "--pitch-model", fcpe]),
        ("fcpe-sequential", ["--pitch", "fcpe", "--pitch-model", fcpe, "--sequential"]),
    ]
    # The same voice with FP16 weight storage (quantize_rvc.py --kind synth): half the download.
    voices = {name: voice for name, _ in cases}
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    try:
        from quantize_rvc import fp16_weights  # noqa: E402

        voice16 = os.path.join(models, "voice_fp16.onnx")
        if not fp16_weights(voice, voice16):
            raise ValueError("voice has no FP32 weights to convert")
        info["voice_fp16"] = run(args.cli, ["inspect", voice16])
        cases.append(("dio-fp16-voice", ["--pitch", "dio"]))
        voices["dio-fp16-voice"] = voice16
    except Exception as e:  # noqa: BLE001 - already FP16, or conversion failed: report, keep going
        print(f"fp16 voice case skipped: {e}")
    failures = 0
    for name, extra in cases:
        out = os.path.join(args.out, f"out-{name}.wav")
        row = {"case": name}
        try:
            stats = run(args.cli, ["convert", "--encoder", enc_q8, "--voice", voices[name], *extra, speech, out])
            y, osr = sf.read(out)
            row.update(stats)
            row["rms"] = float(np.sqrt(np.mean(y ** 2)))
            row["duration_error"] = abs(len(y) / osr - in_dur)
            row["f0_out"] = f0_median(out)
            expect = in_f0 * (2.0 if name.endswith("+12") else 1.0)
            # RVC transposes speaker timbre, not pitch: the median F0 follows the source.
            row["f0_ratio"] = row["f0_out"] / expect if expect else 0
            ok = row["duration_error"] < 0.05 and row["rms"] > 0.005 and np.all(np.isfinite(y)) and 0.8 < row["f0_ratio"] < 1.25
            row["pass"] = bool(ok)
        except Exception as e:  # noqa: BLE001
            row.update({"pass": False, "error": str(e)})
        failures += 0 if row["pass"] else 1
        results.append(row)

    # INT8 analysis on the real encoder: every strategy vs FP32 on the same speech.
    int8 = {}
    try:
        enc_fp32 = hf_file(BASE_REPO, "contentvec_768l12.onnx", models)
        from quantize_rvc import load_audio, parity_encoder, sweep_encoder  # noqa: E402

        audio = load_audio(speech)
        int8["sweep"] = sweep_encoder(enc_fp32, os.path.join(args.out, "sweep"), audio)
        up = parity_encoder(enc_fp32, enc_q8, audio)
        int8["upstream_q8"] = {"variant": "upstream voiceclonnx q8", "mb": round(os.path.getsize(enc_q8) / 1e6, 1),
                               "cos_mean": round(up["cosine_mean"], 4), "cos_p5": round(up["cosine_p5"], 4), "pass": up["pass"]}
        for r in int8["sweep"]:
            r.pop("file", None)
    except Exception as e:  # noqa: BLE001
        int8["error"] = str(e)

    json.dump({"input_seconds": in_dur, "input_f0": in_f0, "models": info, "results": results, "int8": int8},
              open(os.path.join(args.out, "results.json"), "w"), indent=2, default=str)

    lines = ["## RVC engine × real models", "", f"Input: espeak-ng speech, {in_dur:.2f} s, median F0 {in_f0:.0f} Hz", "",
             *([f"Voice: {os.path.getsize(voice) / 1e6:.0f} MB, FP16 conversion {os.path.getsize(voices['dio-fp16-voice']) / 1e6:.0f} MB", ""]
               if "dio-fp16-voice" in voices else []),
             "| case | pass | RTF | encoder ms | pitch ms | synth ms | chunks | out F0 ratio | RMS | note |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['case']} | {'✅' if r['pass'] else '❌'} | {r.get('rtf', 0):.3f} | {r.get('encoderMs', 0):.0f} | "
                     f"{r.get('pitchMs', 0):.0f} | {r.get('synthMs', 0):.0f} | {r.get('chunks', '')} | "
                     f"{r.get('f0_ratio', 0):.2f} | {r.get('rms', 0):.3f} | {r.get('error', '')[:80]} |")
    if "sweep" in int8:
        lines += ["", "### ContentVec quantisation sweep (vs FP32 features on the same speech)", "",
                  "| variant | size MB | mean cos | p5 cos | ms | pass |", "|---|---|---|---|---|---|"]
        for r in [*int8["sweep"], int8.get("upstream_q8", {})]:
            if r:
                lines.append(f"| {r['variant']} | {r.get('mb', '')} | {r.get('cos_mean', 0)} | {r.get('cos_p5', '')} | "
                             f"{r.get('ms', '')} | {'✅' if r.get('pass') else '❌'} {r.get('error', '')} |")
    elif "error" in int8:
        lines += ["", f"INT8 analysis failed: {int8['error']}"]
    report = "\n".join(lines) + "\n"
    open(os.path.join(args.out, "report.md"), "w").write(report)
    print(report)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
