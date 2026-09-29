#!/usr/bin/env python3
"""
MobiGPT — INT8 / FP16 quantisation *with measured parity* for the RVC pipeline.

Blind `quantize_dynamic(model)` on ContentVec is known to hurt intelligibility
badly (voiceclonnx measured WER 38 % -> 69 %). The damage comes from quantising
the convolutional feature extractor and the positional conv, whose activations
have heavy outliers. This tool therefore applies *selective* quantisation per
component and reports numerical parity against the fp32 model so every
shipped file comes with evidence:

  encoder (HuBERT/ContentVec) : INT8 dynamic, MatMul/Gemm only, per-channel
                                -> transformer blocks (~85 % of weights) are
                                   quantised, Conv front-end stays fp32
                                metric: frame-wise cosine similarity
  rmvpe                       : INT8 dynamic on MatMul/Gemm (+ GRU optional)
                                metric: F0 error in cents + voicing agreement
  fcpe                        : INT8 dynamic MatMul/Gemm
                                metric: F0 error in cents
  synth (net_g, per voice)    : FP16 weights with fp32 I/O (INT8 on the HiFi-GAN
                                style decoder is audibly metallic)
                                metric: log-mel spectral distance

Usage
-----
  python quantize_rvc.py analyze  model.onnx
  python quantize_rvc.py quantize --kind encoder model.onnx out_q8.onnx [--audio speech.wav] [--report r.json]
  python quantize_rvc.py quantize --kind synth   voice.onnx voice_fp16.onnx

Requires: numpy, onnx, onnxruntime; optional soundfile, onnxconverter-common (fp16).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort

SR = 16000


# ----------------------------------------------------------------- analysis

def analyze(path: str) -> dict:
    m = onnx.load(path, load_external_data=False)
    ops = collections.Counter(n.op_type for n in m.graph.node)
    init = {i.name: i for i in m.graph.initializer}
    by_op: dict[str, int] = collections.Counter()
    for n in m.graph.node:
        for inp in n.input:
            if inp in init:
                t = init[inp]
                by_op[n.op_type] += int(np.prod(t.dims)) * (4 if t.data_type == onnx.TensorProto.FLOAT else 2)
    total = sum(by_op.values()) or 1
    report = {
        "file_mb": round(os.path.getsize(path) / 1e6, 1),
        "ops": dict(ops.most_common()),
        "weight_share_by_op": {k: round(v / total, 3) for k, v in by_op.most_common()},
        "inputs": [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in m.graph.input],
        "outputs": [(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]) for o in m.graph.output],
        "already_quantized": any(o in ops for o in ("MatMulInteger", "DynamicQuantizeLinear", "QLinearConv")),
    }
    return report


# ---------------------------------------------------------------- test data

def load_audio(path: str | None, seconds: float = 4.0) -> np.ndarray:
    if path:
        import soundfile as sf

        x, sr = sf.read(path, dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != SR:
            n = int(len(x) * SR / sr)
            x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
        return x[: int(seconds * SR)]
    # Synthetic "speech-like" signal: vibrato harmonic stack + formant-ish noise bursts.
    t = np.arange(int(seconds * SR)) / SR
    f0 = 140 + 30 * np.sin(2 * np.pi * 0.7 * t) + 8 * np.sin(2 * np.pi * 5.5 * t)
    ph = 2 * np.pi * np.cumsum(f0) / SR
    x = sum(np.sin(h * ph) / h for h in range(1, 12))
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t) ** 2
    rng = np.random.default_rng(0)
    return (0.15 * x * env + 0.01 * rng.standard_normal(len(t))).astype(np.float32)


def session(path: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def encoder_feed(sess: ort.InferenceSession, audio: np.ndarray) -> dict:
    names = [i.name for i in sess.get_inputs()]
    if "source" in names or len(sess.get_inputs()[0].shape) == 3:
        return {names[0]: audio[None, None, :]}
    feed = {names[0]: audio[None, :]}
    if "attention_mask" in names:
        feed["attention_mask"] = np.ones((1, len(audio)), dtype=np.int64)
    return feed


def log_mel(audio: np.ndarray, fmin: float, htk: bool) -> np.ndarray:
    import librosa

    S = np.abs(librosa.stft(audio, n_fft=1024, hop_length=160, win_length=1024, center=True))
    fb = librosa.filters.mel(sr=SR, n_fft=1024, n_mels=128, fmin=fmin, fmax=8000, htk=htk)
    return np.log(np.maximum(fb @ S, 1e-5)).astype(np.float32)


def decode_salience(sal: np.ndarray, thr: float) -> np.ndarray:
    """Same local-average decode as cpp/rvc/Pitch.cpp (RMVPE flavour)."""
    cents = 20 * np.arange(360) + 1997.3794084376191
    f0 = np.zeros(len(sal), dtype=np.float32)
    for t, row in enumerate(sal):
        c = int(row.argmax())
        if row[c] <= thr:
            continue
        lo, hi = max(0, c - 4), min(360, c + 5)
        w = row[lo:hi]
        f0[t] = 10 * 2 ** ((w * cents[lo:hi]).sum() / w.sum() / 1200)
    return f0


# ------------------------------------------------------------------ parity

def parity_encoder(a: str, b: str, audio: np.ndarray) -> dict:
    sa, sb = session(a), session(b)
    fa = sa.run(None, encoder_feed(sa, audio))[0][0]
    fb = sb.run(None, encoder_feed(sb, audio))[0][0]
    cos = (fa * fb).sum(-1) / (np.linalg.norm(fa, axis=-1) * np.linalg.norm(fb, axis=-1) + 1e-9)
    return {"cosine_mean": float(cos.mean()), "cosine_p5": float(np.percentile(cos, 5)), "frames": int(len(cos)),
            "pass": bool(cos.mean() > 0.97 and np.percentile(cos, 5) > 0.90)}


def parity_pitch(a: str, b: str, audio: np.ndarray, kind: str) -> dict:
    sa, sb = session(a), session(b)
    inp = sa.get_inputs()[0]
    if kind == "rmvpe":
        mel = log_mel(audio, 30, True)
        t = mel.shape[1]
        pad = 32 * ((t - 1) // 32 + 1) - t
        mel = np.pad(mel, ((0, 0), (0, pad)), mode="reflect")[None]
        feed = {inp.name: mel}
        thr = 0.03
    else:
        mel = log_mel(audio, 0, False).T[None]
        feed = {inp.name: mel}
        thr = 0.006
    ya, yb = sa.run(None, feed)[0][0], sb.run(None, feed)[0][0]
    if ya.shape[-1] == 360:
        fa, fb_ = decode_salience(ya, thr), decode_salience(yb, thr)
    else:
        fa, fb_ = ya.reshape(-1), yb.reshape(-1)
    both = (fa > 0) & (fb_ > 0)
    cents = np.abs(1200 * np.log2(np.maximum(fb_[both], 1) / np.maximum(fa[both], 1))) if both.any() else np.array([0.0])
    vuv = float(((fa > 0) == (fb_ > 0)).mean())
    return {"cents_mean": float(cents.mean()), "cents_p95": float(np.percentile(cents, 95)), "voicing_agreement": vuv,
            "pass": bool(np.percentile(cents, 95) < 50 and vuv > 0.95)}


def parity_synth(a: str, b: str, frames: int = 200) -> dict:
    sa, sb = session(a), session(b)
    rng = np.random.default_rng(1)
    feed = {}
    for i in sa.get_inputs():
        shape = [frames if isinstance(d, str) or d in (None, -1) else d for d in i.shape]
        if i.name in ("phone", "feats"):
            ch = shape[2] if isinstance(shape[2], int) and shape[2] > 0 else 768
            feed[i.name] = rng.standard_normal((1, frames, ch)).astype(np.float32) * 0.3
        elif i.name in ("phone_lengths", "p_len"):
            feed[i.name] = np.array([frames], dtype=np.int64)
        elif i.name == "pitch":
            feed[i.name] = np.full((1, frames), 60, dtype=np.int64)
        elif i.name == "pitchf":
            feed[i.name] = np.full((1, frames), 180.0, dtype=np.float32)
        elif i.name in ("ds", "sid"):
            feed[i.name] = np.array([0], dtype=np.int64)
        elif i.name == "rnd":
            feed[i.name] = rng.standard_normal((1, 192, frames)).astype(np.float32)
    if "float16" in sa.get_inputs()[0].type:
        feed[sa.get_inputs()[0].name] = feed[sa.get_inputs()[0].name].astype(np.float16)
    ya = sa.run(None, feed)[0].reshape(-1).astype(np.float32)
    yb = sb.run(None, feed)[0].reshape(-1).astype(np.float32)
    n = min(len(ya), len(yb))
    ma, mb = log_mel(ya[:n], 0, False), log_mel(yb[:n], 0, False)
    dist = float(np.sqrt(((ma - mb) ** 2).mean()))
    return {"log_mel_rmse": dist, "pass": bool(dist < 0.5)}


# ---------------------------------------------------------------- quantise

def quantize(kind: str, src: str, dst: str, ops: list[str] | None) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    if kind == "synth":
        from onnxconverter_common import float16

        m = onnx.load(src)
        m16 = float16.convert_float_to_float16(m, keep_io_types=True, op_block_list=["RandomNormalLike", "Range", "CumSum"])
        onnx.save(m16, dst)
        return
    default_ops = {"encoder": ["MatMul", "Gemm"], "rmvpe": ["MatMul", "Gemm"], "fcpe": ["MatMul", "Gemm"]}[kind]
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8, per_channel=True, reduce_range=False,
                     op_types_to_quantize=ops or default_ops)


def bench(path: str, feed_fn, runs: int = 3) -> float:
    s = session(path)
    feed = feed_fn(s)
    s.run(None, feed)
    t0 = time.perf_counter()
    for _ in range(runs):
        s.run(None, feed)
    return (time.perf_counter() - t0) / runs * 1000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("model")
    q = sub.add_parser("quantize")
    q.add_argument("--kind", required=True, choices=["encoder", "rmvpe", "fcpe", "synth"])
    q.add_argument("--ops", nargs="*", help="override op types to quantise (e.g. MatMul Gemm GRU)")
    q.add_argument("--audio", help="speech WAV for parity (default: synthetic)")
    q.add_argument("--report", help="write JSON report here")
    q.add_argument("src")
    q.add_argument("dst")
    args = ap.parse_args()

    if args.cmd == "analyze":
        print(json.dumps(analyze(args.model), indent=2))
        return 0

    quantize(args.kind, args.src, args.dst, args.ops)
    audio = load_audio(args.audio)
    report = {"kind": args.kind, "src_mb": round(os.path.getsize(args.src) / 1e6, 1), "dst_mb": round(os.path.getsize(args.dst) / 1e6, 1)}
    if args.kind == "encoder":
        report["parity"] = parity_encoder(args.src, args.dst, audio)
        feed = lambda s: encoder_feed(s, audio)  # noqa: E731
        report["ms_fp32"], report["ms_quant"] = bench(args.src, feed), bench(args.dst, feed)
    elif args.kind in ("rmvpe", "fcpe"):
        report["parity"] = parity_pitch(args.src, args.dst, audio, args.kind)
    else:
        report["parity"] = parity_synth(args.src, args.dst)
    report["size_reduction"] = round(1 - report["dst_mb"] / report["src_mb"], 3)
    print(json.dumps(report, indent=2))
    if args.report:
        json.dump(report, open(args.report, "w"), indent=2)
    return 0 if report["parity"]["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
