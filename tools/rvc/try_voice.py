#!/usr/bin/env python3
"""Can MobiGPT run this RVC voice? Converts it if needed and runs real speech through the engine.

Input: an .onnx voice, a .pth, or a .zip containing a .pth (the usual community format,
often with a FAISS .index that MobiGPT does not need).

Usage: python tools/rvc/try_voice.py --input VOICE --rvc-repo RVC --cli mobigpt-rvc --out DIR
"""
import argparse
import json
import os
import subprocess
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def extract_voice(path: str, out: str) -> tuple[str, str]:
    """Returns (path, kind) where kind is 'onnx' or 'pth'."""
    raw = open(path, "rb").read(8)
    if raw[:4] != b"PK\x03\x04":
        return path, "onnx"
    z = zipfile.ZipFile(path)
    if any(n.endswith("data.pkl") for n in z.namelist()):
        return path, "pth"
    for kind in ("onnx", "pth"):
        for n in z.namelist():
            if n.lower().endswith("." + kind):
                dst = os.path.join(out, f"voice.{kind}")
                open(dst, "wb").write(z.read(n))
                return dst, kind
    raise SystemExit(f"no .pth or .onnx inside {path}: {z.namelist()}")


def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout).strip()[-2000:])
    return p.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--rvc-repo", required=True)
    ap.add_argument("--cli", required=True)
    ap.add_argument("--out", default="voice-try")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    voice, kind = extract_voice(args.input, args.out)
    lines = ["### Running the voice through MobiGPT's engine", ""]
    device_voice = None
    if kind == "pth":
        # The app's own path: native .zip/.pth import into a weight-free template, on the phone.
        from make_voice_templates import export_template

        rinfo = json.loads(run([args.cli, "rvc-info", args.input]).strip().splitlines()[-1])
        tpl = os.path.join(args.out, rinfo["template"])
        export_template(args.rvc_repo, rinfo["version"], rinfo["sampleRate"], tpl)
        imported = os.path.join(args.out, "device-import")
        os.makedirs(imported, exist_ok=True)
        run([args.cli, "rvc-import", "--template", tpl, args.input, imported])
        device_voice = os.path.join(imported, "model.onnx")
        lines += [f"* On-device import (as the app does it): `{rinfo['pth']}` → {rinfo['template']} + "
                  f"{os.path.getsize(os.path.join(imported, 'weights.bin')) / 1e6:.0f} MB weights"
                  f"{' · bundled .index not needed' if rinfo['hasIndex'] else ''}"]
        onnx_path = os.path.join(args.out, "voice.onnx")
        out = run([sys.executable, os.path.join(HERE, "export_voice_onnx.py"), "--rvc-repo", args.rvc_repo, "--fp16", voice, onnx_path])
        lines += [f"* Converted `.pth` → ONNX: {out.strip().splitlines()[-1]}"]
        voice = onnx_path
    info = json.loads(run([args.cli, "inspect", voice]).strip().splitlines()[-1])
    lines += [f"* Engine sees: **{info['kind']}**, {info.get('layout')} {info.get('channels')}-d, "
              f"{info.get('sampleRate')} Hz, f0={info.get('f0')}, {info['sizeMB']:.0f} MB"]

    from ci_real_models import BASE_REPO, f0_median, hf_file

    models = os.path.join(args.out, "models")
    enc = hf_file(BASE_REPO, "contentvec_768l12_q8.onnx", models) if info.get("channels") != 256 else None
    if enc is None:
        lines += ["* v1 (256-d) voice: needs the HuBERT 256-d encoder; skipping the run"]
    speech = os.path.join(args.out, "speech.wav")
    subprocess.run(["espeak-ng", "-v", "en-us", "-s", "150", "-w", speech,
                    "Private artificial intelligence runs entirely on this phone. Nothing leaves the device."], check=True)
    import soundfile as sf

    x, sr = sf.read(speech)
    in_f0 = f0_median(speech)
    ok = True
    if enc:
        fcpe = hf_file("niobures/FCPE", "onnx/fcpe.onnx", models)
        lines += ["", "| pitch | RTF | out F0 / in F0 | RMS | result |", "|---|---|---|---|---|"]
        cases = [("dio", voice, ["--pitch", "dio"]), ("fcpe", voice, ["--pitch", "fcpe", "--pitch-model", fcpe]),
                 ("fcpe +12", voice, ["--pitch", "fcpe", "--pitch-model", fcpe, "--key", "12"])]
        if device_voice:
            cases.append(("fcpe · on-device import", device_voice, ["--pitch", "fcpe", "--pitch-model", fcpe]))
        for name, model, extra in cases:
            out_wav = os.path.join(args.out, f"out-{name.split(' ')[0]}{'-imported' if model == device_voice else ''}{'+12' if '+12' in name else ''}.wav")
            try:
                st = json.loads(run([args.cli, "convert", "--encoder", enc, "--voice", model, *extra, speech, out_wav]).strip().splitlines()[-1])
                y, osr = sf.read(out_wav)
                rms = float(np.sqrt(np.mean(y ** 2)))
                ratio = f0_median(out_wav) / (in_f0 * (2 if "+12" in name else 1))
                good = rms > 0.005 and abs(len(y) / osr - len(x) / sr) < 0.05 and 0.8 < ratio < 1.25
                ok &= good
                lines.append(f"| {name} | {st['rtf']:.2f} | {ratio:.2f} | {rms:.3f} | {'✅' if good else '❌'} |")
            except Exception as e:  # noqa: BLE001
                ok = False
                lines.append(f"| {name} | | | | ❌ {str(e)[:120]} |")
    lines += ["", "**Verdict:** " + ("works in MobiGPT ✅" if ok else "does not work as is ❌")]
    report = "\n".join(lines) + "\n"
    open(os.path.join(args.out, "report.md"), "w").write(report)
    print(report)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
