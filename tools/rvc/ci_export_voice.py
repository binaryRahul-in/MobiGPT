#!/usr/bin/env python3
"""CI check for export_voice_onnx.py against the real RVC model code.

Builds a randomly initialised RVC v2 40 kHz voice checkpoint from the pinned
RVC-WebUI checkout, exports it twice (FP32 and --fp16), and checks that both load,
have the RVC-WebUI I/O contract and metadata, and that the FP16-weight export
sounds the same as FP32 (audible-band mel error at the SineGen noise floor).

Usage: python tools/rvc/ci_export_voice.py --rvc-repo rvc --out build/voice-export
Requires: torch, onnx, onnxruntime, librosa.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def make_checkpoint(rvc_repo: str, path: str) -> None:
    import torch

    sys.path.insert(0, os.path.abspath(rvc_repo))
    from infer.lib.infer_pack.models import SynthesizerTrnMs768NSFsid  # type: ignore

    d = json.load(open(os.path.join(rvc_repo, "configs/v1/40k.json")))  # RVC's v2 40k voices use this config
    m = d["model"]
    cfg = [d["data"]["filter_length"] // 2 + 1, d["train"]["segment_size"] // d["data"]["hop_length"],
           m["inter_channels"], m["hidden_channels"], m["filter_channels"], m["n_heads"], m["n_layers"], m["kernel_size"],
           m["p_dropout"], m["resblock"], m["resblock_kernel_sizes"], m["resblock_dilation_sizes"], m["upsample_rates"],
           m["upsample_initial_channel"], m["upsample_kernel_sizes"], m["spk_embed_dim"], m["gin_channels"], 40000]
    torch.manual_seed(0)
    net = SynthesizerTrnMs768NSFsid(*cfg, is_half=False)
    weights = {k: v for k, v in net.state_dict().items() if "enc_q" not in k}  # inference checkpoints drop enc_q
    torch.save({"weight": weights, "config": cfg, "version": "v2", "f0": 1}, path)


def audible_db_error(a: np.ndarray, b: np.ndarray) -> float:
    import librosa

    A = librosa.power_to_db(librosa.feature.melspectrogram(y=a, sr=40000, n_mels=80), ref=1.0)
    B = librosa.power_to_db(librosa.feature.melspectrogram(y=b, sr=40000, n_mels=80), ref=1.0)
    mask = A > A.max() - 60
    return float(np.sqrt(np.mean((A[mask] - B[mask]) ** 2)))


def main() -> int:
    import onnx
    import onnxruntime as ort

    ap = argparse.ArgumentParser()
    ap.add_argument("--rvc-repo", required=True)
    ap.add_argument("--out", default="voice-export")
    ap.add_argument("--cli", help="mobigpt-rvc binary: also check the on-device .zip/.pth importer")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pth = os.path.join(args.out, "random_v2_40k.pth")
    make_checkpoint(args.rvc_repo, pth)
    paths = {}
    for label, extra in (("fp32", []), ("fp16", ["--fp16"])):
        paths[label] = os.path.join(args.out, f"voice_{label}.onnx")
        subprocess.run([sys.executable, os.path.join(HERE, "export_voice_onnx.py"), "--rvc-repo", args.rvc_repo, *extra, pth,
                        paths[label]], check=True)

    rng = np.random.default_rng(0)
    T = 250
    feed = {"phone": rng.standard_normal((1, T, 768)).astype(np.float32), "phone_lengths": np.array([T], np.int64),
            "pitch": np.full((1, T), 60, np.int64), "pitchf": np.full((1, T), 220.0, np.float32),
            "ds": np.array([0], np.int64), "rnd": rng.standard_normal((1, 192, T)).astype(np.float32)}
    out, failures = {}, []
    for label, p in paths.items():
        model = onnx.load(p, load_external_data=False)
        meta = {x.key: x.value for x in model.metadata_props}
        s = ort.InferenceSession(p)
        names = [i.name for i in s.get_inputs()]
        if names != list(feed):
            failures.append(f"{label}: inputs {names}")
        if meta.get("sample_rate") != "40000" or meta.get("version") != "v2" or meta.get("f0") != "1":
            failures.append(f"{label}: metadata {meta}")
        y = s.run(None, feed)[0]
        if y.shape != (1, 1, T * 400) or y.dtype != np.float32 or not np.isfinite(y).all():
            failures.append(f"{label}: output {y.shape} {y.dtype}")
        out[label] = y.ravel()
        print(f"{label}: {os.path.getsize(p) / 1e6:.1f} MB, inputs ok, metadata {meta}")

    noise = audible_db_error(out["fp32"], ort.InferenceSession(paths["fp32"]).run(None, feed)[0].ravel())
    err = audible_db_error(out["fp32"], out["fp16"])
    ratio = os.path.getsize(paths["fp16"]) / os.path.getsize(paths["fp32"])
    print(f"FP16-weight export: {ratio:.2f}x size, audible-band error {err:.2f} dB (FP32 run-to-run {noise:.2f} dB)")
    if ratio > 0.6:
        failures.append(f"fp16 export is {ratio:.2f}x the FP32 size")
    # Weight rounding measures 0.1-0.4 dB depending on the SineGen noise draw; full FP16 compute is ~4.6 dB.
    if err > max(1.0, 3 * noise):
        failures.append(f"fp16 export error {err:.2f} dB")
    if args.cli:
        failures += check_device_import(args, pth, feed, out["fp16"])
    for f in failures:
        print("FAIL", f)
    return 1 if failures else 0


def check_device_import(args, pth: str, feed: dict, reference: np.ndarray) -> list[str]:
    """The app's native importer (template + .pth inside an RVC .zip) must reproduce the export."""
    import zipfile

    import onnxruntime as ort
    import torch

    sys.path.insert(0, HERE)
    from make_voice_templates import export_template

    # What RVC's "save small model" writes: FP16 weights without enc_q, plus config and tags.
    ck = torch.load(pth, map_location="cpu", weights_only=False)
    small = {"weight": {k: v.half() for k, v in ck["weight"].items() if "enc_q" not in k}, "config": ck["config"],
             "info": "ci", "sr": "40k", "f0": 1, "version": "v2"}
    small_pth = os.path.join(args.out, "Random Voice.pth")
    torch.save(small, small_pth)
    bundle = os.path.join(args.out, "Random Voice.zip")
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(small_pth, "Random Voice/Random Voice.pth")
        z.writestr("Random Voice/added_IVF1_Flat_nprobe_1_Random_Voice_v2.index", b"\0" * 4096)
    template = os.path.join(args.out, "rvc_template_v2_40k.onnx")
    info = export_template(args.rvc_repo, "v2", 40000, template)
    imported = os.path.join(args.out, "imported")
    os.makedirs(imported, exist_ok=True)
    res = subprocess.run([args.cli, "rvc-import", "--template", template, bundle, imported], capture_output=True, text=True)
    if res.returncode != 0:
        return [f"on-device import failed: {res.stderr.strip()}"]
    print("on-device import:", res.stdout.strip(), f"(template {info['bytes'] / 1e6:.1f} MB)")
    y = ort.InferenceSession(os.path.join(imported, "model.onnx")).run(None, feed)[0].ravel()
    corr = float(np.corrcoef(reference, y)[0, 1])
    err = audible_db_error(reference, y)
    print(f"imported vs export --fp16: corr {corr:.5f}, audible-band error {err:.2f} dB")
    return [] if corr > 0.999 and err < 1.0 else [f"imported voice differs from the export: corr {corr:.5f}, {err:.2f} dB"]


if __name__ == "__main__":
    sys.exit(main())
