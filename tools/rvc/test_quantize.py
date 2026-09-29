#!/usr/bin/env python3
"""Regression tests for quantize_rvc.py: every RVC model layout survives FP16 and INT8 conversion.

Builds the engine's test fixtures (all encoder, pitch and synthesiser I/O layouts),
converts each one, loads it in ONNX Runtime and compares outputs with the original.

Usage: python tools/rvc/test_quantize.py
Requires: numpy, onnx, onnxruntime, onnxconverter-common (librosa optional).
"""
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper as nh

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import quantize_rvc as q  # noqa: E402

rng = np.random.default_rng(0)


def feed(sess: ort.InferenceSession, n_audio: int = 3200, frames: int = 64) -> dict:
    out = {}
    for i in sess.get_inputs():
        is_audio = i.name in ("source", "input_values", "audio", "waveform")
        dims = [d if isinstance(d, int) else (n_audio if is_audio else frames) for d in i.shape]
        if len(dims) > 1 and not isinstance(i.shape[0], int):
            dims[0] = 1
        if "int64" in i.type:
            v = frames if ("length" in i.name or i.name == "p_len") else 1
            out[i.name] = np.full(dims, v, np.int64)
        elif "float16" in i.type:
            out[i.name] = (rng.random(dims) - 0.5).astype(np.float16)
        else:
            val = rng.random(dims).astype(np.float32)
            out[i.name] = val * 400 + 100 if i.name == "pitchf" else val - 0.5
    return out


def rel_err(a: list, b: list) -> float:
    return max(float(np.abs(x.astype(np.float32) - y.astype(np.float32)).max() / (np.abs(x).max() + 1e-9)) for x, y in zip(a, b))


def check_pair(src: str, dst: str, tol: float) -> float:
    a, b = ort.InferenceSession(src), ort.InferenceSession(dst)
    assert [i.type for i in a.get_inputs()] == [i.type for i in b.get_inputs()], "graph input types changed"
    x = feed(a)
    err = rel_err(a.run(None, x), b.run(None, x))
    assert err < tol, f"relative error {err:.3g} >= {tol}"
    return err


def cast_mask_model(path: str) -> None:
    """HuBERT-style attention-mask Cast(to=FLOAT) plus a graph output consumed internally."""
    w = nh.from_array(rng.standard_normal((8, 4)).astype(np.float32), "W")
    w2 = nh.from_array(rng.standard_normal((4, 4)).astype(np.float32), "W2")
    zero = nh.from_array(np.zeros((1,), np.float32), "zero")
    nodes = [
        h.make_node("Greater", ["x", "zero"], ["g"], name="gt"),
        h.make_node("Cast", ["g"], ["m"], to=T.FLOAT, name="/hubert/encoder/Cast_1"),
        h.make_node("Mul", ["x", "m"], ["xm"], name="mul"),
        h.make_node("MatMul", ["xm", "W"], ["y"], name="mm"),
        h.make_node("MatMul", ["y", "W2"], ["z"], name="mm2"),
    ]
    g = h.make_graph(nodes, "g", [h.make_tensor_value_info("x", T.FLOAT, [1, 3, 8])],
                     [h.make_tensor_value_info("y", T.FLOAT, [1, 3, 4]), h.make_tensor_value_info("z", T.FLOAT, [1, 3, 4])],
                     [w, w2, zero])
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(onnx.shape_inference.infer_shapes(m), path)


def main() -> int:
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        fx = os.path.join(tmp, "fixtures")
        subprocess.run([sys.executable, os.path.join(ROOT, "packages/react-native-mobigpt-voice/tools/make_test_fixtures.py"), fx],
                       check=True, stdout=subprocess.DEVNULL)
        cast_mask_model(os.path.join(fx, "cast_mask.onnx"))

        for src in sorted(glob.glob(os.path.join(fx, "*.onnx"))):
            name = os.path.basename(src)
            cases = [("fp16", 5e-2)]
            if name.startswith(("enc_", "rmvpe_", "fcpe", "cast_")):
                cases.append(("int8", 0.2))
            for kind, tol in cases:
                dst = os.path.join(tmp, f"{kind}_{name}")
                try:
                    if kind == "fp16":
                        # voices go through the production path, which keeps phase accumulation in FP32
                        q.quantize("synth", src, dst, None) if name.startswith("synth_") else q.to_fp16(src, dst)
                    else:
                        q.quantize("encoder", src, dst, None)
                    err = check_pair(src, dst, tol)
                    print(f"ok    {kind:5} {name:32} rel err {err:.1e}")
                except ValueError as e:
                    if "fp16" in name:  # already FP16: refusing is the correct outcome
                        print(f"skip  {kind:5} {name:32} {e}")
                    else:
                        failures.append(f"{kind} {name}: {e}")
                except Exception as e:  # noqa: BLE001
                    failures.append(f"{kind} {name}: {str(e)[:300]}")
                    print(f"FAIL  {kind:5} {name:32} {str(e)[:300]}")
    if failures:
        print(f"\n{len(failures)} failure(s)")
        return 1
    print("\nall conversions load and match the originals")
    return 0


if __name__ == "__main__":
    sys.exit(main())
