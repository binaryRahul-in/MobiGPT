#!/usr/bin/env python3
"""Generate tiny ONNX models + reference data for the C++ unit tests.

The models reproduce the exact I/O contracts of real RVC exports (RVC-WebUI,
transformers/ContentVec and w-okada voice-changer layouts, fp32 and fp16) but
are only a few MB, so CI can exercise the whole pipeline without downloading
~800 MB of weights. The synthesiser is a real (if trivial) vocoder: it renders
a phase-continuous sine at `pitchf`, which lets tests verify pitch shifting,
chunking and SOLA stitching numerically.

Usage: python make_test_fixtures.py <out_dir>
Requires: numpy, onnx (and librosa for the mel reference data).
"""
import json
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 17
rng = np.random.default_rng(0)


def save(model, path, meta=None, check=True):
    if meta:
        for k, v in meta.items():
            p = model.metadata_props.add()
            p.key, p.value = k, v
    model.opset_import[0].version = OPSET
    model.ir_version = 8
    if check:  # templates reference a weights.bin that only exists after an import
        onnx.checker.check_model(model)
    onnx.save(model, path)


def const(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name)


# --------------------------------------------------------------- encoders

def encoder(path, layout, channels=768):
    """Conv1d(k=400, s=320) front-end, like HuBERT's feature extractor."""
    w = (rng.standard_normal((channels, 1, 400)) * 0.01).astype(np.float32)
    inits = [const("W", w)]
    nodes = []
    if layout == "webui":
        inp = [helper.make_tensor_value_info("source", TensorProto.FLOAT, [1, 1, "N"])]
        src = "source"
        out_name = "embed"
    elif layout == "hf":
        inp = [
            helper.make_tensor_value_info("input_values", TensorProto.FLOAT, [1, "N"]),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [1, "N"]),
        ]
        inits.append(const("ax1", np.array([1], dtype=np.int64)))
        nodes.append(helper.make_node("Unsqueeze", ["input_values", "ax1"], ["src3"]))
        src = "src3"
        out_name = "hidden_states"
    else:  # wokada
        inp = [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, "N"])]
        inits.append(const("ax1", np.array([1], dtype=np.int64)))
        nodes.append(helper.make_node("Unsqueeze", ["audio", "ax1"], ["src3"]))
        src = "src3"
        out_name = "unit12"
    nodes += [
        helper.make_node("Conv", [src, "W"], ["conv"], strides=[320]),
        helper.make_node("Tanh", ["conv"], ["act"]),
        helper.make_node("Transpose", ["act"], [out_name], perm=[0, 2, 1]),
    ]
    outs = [helper.make_tensor_value_info(out_name, TensorProto.FLOAT, [1, "F", channels])]
    if layout == "wokada":
        w9 = (rng.standard_normal((channels, 256)) * 0.05).astype(np.float32)
        inits.append(const("W9", w9))
        nodes += [
            helper.make_node("MatMul", ["unit12", "W9"], ["units9"]),
            helper.make_node("Identity", ["unit12"], ["unit12s"]),
        ]
        outs += [
            helper.make_tensor_value_info("units9", TensorProto.FLOAT, [1, "F", 256]),
            helper.make_tensor_value_info("unit12s", TensorProto.FLOAT, [1, "F", channels]),
        ]
    g = helper.make_graph(nodes, "encoder", inp, outs, inits)
    save(helper.make_model(g), path)


# ----------------------------------------------------------- pitch models

def rmvpe_mel(path):
    w = (rng.standard_normal((128, 360)) * 0.1).astype(np.float32)
    g = helper.make_graph(
        [
            helper.make_node("Transpose", ["input"], ["t"], perm=[0, 2, 1]),
            helper.make_node("MatMul", ["t", "W"], ["logits"]),
            helper.make_node("Sigmoid", ["logits"], ["hidden"]),
        ],
        "rmvpe",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 128, "T"])],
        [helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [1, "T", 360])],
        [const("W", w)],
    )
    save(helper.make_model(g), path)


def rmvpe_wave(path, hz=220.0):
    """w-okada style: waveform + threshold -> pitchf (constant `hz`)."""
    g = helper.make_graph(
        [
            helper.make_node("Slice", ["waveform", "st", "en", "ax", "step"], ["dec"]),
            helper.make_node("Mul", ["dec", "zero"], ["z"]),
            helper.make_node("Add", ["z", "hz"], ["z2"]),
            helper.make_node("Mul", ["threshold", "zero1"], ["tz"]),
            helper.make_node("Add", ["z2", "tz"], ["pitchf"]),
        ],
        "rmvpe_wave",
        [
            helper.make_tensor_value_info("waveform", TensorProto.FLOAT, [1, "N"]),
            helper.make_tensor_value_info("threshold", TensorProto.FLOAT, [1]),
        ],
        [helper.make_tensor_value_info("pitchf", TensorProto.FLOAT, [1, "T"])],
        [
            const("st", np.array([0], dtype=np.int64)),
            const("en", np.array([2**62], dtype=np.int64)),
            const("ax", np.array([1], dtype=np.int64)),
            const("step", np.array([160], dtype=np.int64)),
            const("zero", np.array(0.0, dtype=np.float32)),
            const("zero1", np.array([0.0], dtype=np.float32)),
            const("hz", np.array(hz, dtype=np.float32)),
        ],
    )
    save(helper.make_model(g), path)


def fcpe(path):
    w = (rng.standard_normal((128, 360)) * 0.1).astype(np.float32)
    g = helper.make_graph(
        [helper.make_node("MatMul", ["mel", "W"], ["l"]), helper.make_node("Sigmoid", ["l"], ["latent"])],
        "fcpe",
        [helper.make_tensor_value_info("mel", TensorProto.FLOAT, [1, "T", 128])],
        [helper.make_tensor_value_info("latent", TensorProto.FLOAT, [1, "T", 360])],
        [const("W", w)],
    )
    save(helper.make_model(g), path)


def mel_threshold(path, layout="fcpe"):
    """Pitch models that decode inside the graph and take the voicing threshold as a
    second input (niobures/FCPE): mel + threshold -> Hz. Voiced (220 Hz) iff
    threshold <= 0.5, so a missing or wrong threshold input is detected."""
    mel_shape = [1, "T", 128] if layout == "fcpe" else [1, 128, "T"]
    reduce_axis = 2 if layout == "fcpe" else 1
    g = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["mel"], ["m"], keepdims=0, axes=[reduce_axis]),  # [1,T]
            helper.make_node("Mul", ["m", "zero"], ["z"]),
            helper.make_node("Add", ["z", "hz"], ["voiced"]),
            helper.make_node("LessOrEqual", ["threshold", "half"], ["ok"]),
            helper.make_node("Where", ["ok", "voiced", "z"], ["f0"]),
        ],
        "mel_threshold",
        [helper.make_tensor_value_info("mel", TensorProto.FLOAT, mel_shape),
         helper.make_tensor_value_info("threshold", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("f0", TensorProto.FLOAT, [1, "T"])],
        [const("zero", np.array(0, dtype=np.float32)), const("hz", np.array(220, dtype=np.float32)),
         const("half", np.array([0.5], dtype=np.float32))],
    )
    save(helper.make_model(g), path)


# ------------------------------------------------------------ synthesisers

def synth(path, layout, sr, fp16=False, with_f0=True, channels=768, meta=None):
    hop = sr // 100
    feats_name = "phone" if layout == "webui" else "feats"
    len_name = "phone_lengths" if layout == "webui" else "p_len"
    sid_name = "ds" if layout == "webui" else "sid"
    ftype = TensorProto.FLOAT16 if fp16 else TensorProto.FLOAT
    inputs = [
        helper.make_tensor_value_info(feats_name, ftype, [1, "T", channels]),
        helper.make_tensor_value_info(len_name, TensorProto.INT64, [1]),
    ]
    if with_f0:
        inputs += [
            helper.make_tensor_value_info("pitch", TensorProto.INT64, [1, "T"]),
            helper.make_tensor_value_info("pitchf", TensorProto.FLOAT, [1, "T"]),
        ]
    inputs.append(helper.make_tensor_value_info(sid_name, TensorProto.INT64, [1]))
    if layout == "webui":
        inputs.append(helper.make_tensor_value_info("rnd", TensorProto.FLOAT, [1, 192, "T"]))

    nodes, inits = [], [
        const("two_pi_over_sr", np.array(2 * np.pi / sr, dtype=np.float32)),
        const("amp", np.array(0.3, dtype=np.float32)),
        const("tiny", np.array(1e-6, dtype=np.float32)),
        const("rep", np.array([1, 1, hop], dtype=np.int64)),
        const("flat", np.array([1, 1, -1], dtype=np.int64)),
        const("ax2", np.array([2], dtype=np.int64)),
        const("ax_last", np.array(2, dtype=np.int64)),
        const("tone", np.array(300.0, dtype=np.float32)),
    ]
    feats = feats_name
    if fp16:
        nodes.append(helper.make_node("Cast", [feats_name], ["feats32"], to=TensorProto.FLOAT))
        feats = "feats32"
    # Tie every input into the graph (weight ~0) so ORT keeps them.
    nodes.append(helper.make_node("ReduceMean", [feats], ["fmean"], keepdims=1, axes=[2]))  # [1,T,1]
    if with_f0:
        nodes += [helper.make_node("Unsqueeze", ["pitchf", "ax2"], ["f3"])]  # [1,T,1]
    else:
        nodes += [helper.make_node("Mul", ["fmean", "tiny"], ["fm0"]), helper.make_node("Add", ["fm0", "tone"], ["f3"])]
    nodes += [
        helper.make_node("Tile", ["f3", "rep"], ["f_rep"]),        # [1,T,hop]
        helper.make_node("Reshape", ["f_rep", "flat"], ["f_s"]),     # [1,1,T*hop]
        helper.make_node("Mul", ["f_s", "two_pi_over_sr"], ["dphi"]),
        helper.make_node("CumSum", ["dphi", "ax_last"], ["phi"]),
        helper.make_node("Sin", ["phi"], ["s"]),
        helper.make_node("Mul", ["s", "amp"], ["y0"]),
        helper.make_node("Mul", ["fmean", "tiny"], ["fm_t"]),
        helper.make_node("Tile", ["fm_t", "rep"], ["fm_rep"]),
        helper.make_node("Reshape", ["fm_rep", "flat"], ["fm_s"]),
        helper.make_node("Add", ["y0", "fm_s"], ["y1"]),
    ]
    out = "y1"
    if layout == "webui":
        nodes += [
            helper.make_node("ReduceMean", ["rnd"], ["rmean"], keepdims=1, axes=[1, 2]),  # [1,1,1]
            helper.make_node("Mul", ["rmean", "tiny"], ["rm"]),
            helper.make_node("Add", [out, "rm"], ["y2"]),
        ]
        out = "y2"
    if fp16:
        nodes.append(helper.make_node("Cast", [out], ["audio"], to=TensorProto.FLOAT16))
    else:
        nodes.append(helper.make_node("Identity", [out], ["audio"]))
    g = helper.make_graph(nodes, "synth", inputs, [helper.make_tensor_value_info("audio", ftype, [1, 1, "S"])], inits)
    save(helper.make_model(g), path, meta)


# ------------------------------------------------------------ mel reference

def mel_reference(out):
    import librosa

    sr = 16000
    t = np.arange(8000) / sr
    y = (0.5 * np.sin(2 * np.pi * (120 + 400 * t) * t) + 0.05 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)
    y.tofile(os.path.join(out, "mel_input.f32"))

    def write(name, arr):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        with open(os.path.join(out, name), "wb") as f:
            np.array(arr.shape, dtype=np.int32).tofile(f)
            arr.tofile(f)

    fb_htk = librosa.filters.mel(sr=sr, n_fft=1024, n_mels=128, fmin=30, fmax=8000, htk=True)
    fb_sl = librosa.filters.mel(sr=sr, n_fft=1024, n_mels=128, fmin=0, fmax=8000, htk=False)
    write("fb_htk.bin", fb_htk)
    write("fb_slaney.bin", fb_sl)
    # RMVPE: torch.stft(center=True, reflect) == librosa.stft(center=True, pad_mode="reflect")
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=160, win_length=1024, window="hann", center=True, pad_mode="reflect"))
    write("mel_rmvpe.bin", np.log(np.maximum(fb_htk @ S, 1e-5)))
    # FCPE: manual reflect pad then center=False, sqrt(|X|^2 + 1e-9)
    yp = np.pad(y, (432, 432), mode="reflect")
    X = librosa.stft(yp, n_fft=1024, hop_length=160, win_length=1024, window="hann", center=False)
    S2 = np.sqrt(np.abs(X) ** 2 + 1e-9)
    write("mel_fcpe.bin", np.log(np.maximum(fb_sl @ S2, 1e-5)))


def torch_pth(path, tensors, extra):
    """Writes a real torch.save()-format checkpoint (ZIP: archive/data.pkl + archive/data/<k>)
    without PyTorch. Stand-in `torch` modules make pickle emit exactly the GLOBAL /
    BINPERSID / REDUCE sequence torch uses, so the C++ reader is tested on the real format."""
    import pickle
    import sys
    import types
    import zipfile
    from collections import OrderedDict

    torch_mod = sys.modules.get("torch") or types.ModuleType("torch")
    utils_mod = sys.modules.get("torch._utils") or types.ModuleType("torch._utils")

    def _rebuild_tensor_v2(*a):  # never called; only its name is pickled
        raise RuntimeError

    _rebuild_tensor_v2.__module__, _rebuild_tensor_v2.__qualname__ = "torch._utils", "_rebuild_tensor_v2"
    storage_types = {}
    for nm in ("HalfStorage", "FloatStorage"):
        cls = type(nm, (), {"__module__": "torch"})
        storage_types[nm] = cls
    saved = {k: sys.modules.get(k) for k in ("torch", "torch._utils")}
    fake_torch, fake_utils = types.ModuleType("torch"), types.ModuleType("torch._utils")
    fake_utils._rebuild_tensor_v2 = _rebuild_tensor_v2
    for nm, cls in storage_types.items():
        setattr(fake_torch, nm, cls)
    sys.modules["torch"], sys.modules["torch._utils"] = fake_torch, fake_utils

    class Storage:
        def __init__(self, key, arr):
            self.key, self.arr = key, arr

    class Tensor:
        def __init__(self, storage, shape):
            self.storage, self.shape = storage, shape

        def __reduce__(self):
            stride, acc = [], 1
            for d in reversed(self.shape):
                stride.insert(0, acc)
                acc *= d
            return (_rebuild_tensor_v2, (self.storage, 0, tuple(self.shape), tuple(stride), False, OrderedDict()))

    class Pickler(pickle.Pickler):
        def persistent_id(self, obj):
            if isinstance(obj, Storage):
                nm = "HalfStorage" if obj.arr.dtype == np.float16 else "FloatStorage"
                return ("storage", storage_types[nm], obj.key, "cpu", obj.arr.size)
            return None

    try:
        weight, blobs = OrderedDict(), {}
        for i, (k, arr) in enumerate(tensors.items()):
            st = Storage(str(i), np.ascontiguousarray(arr))
            blobs[st.key] = st.arr.tobytes()
            weight[k] = Tensor(st, list(arr.shape))
        import io

        buf = io.BytesIO()
        Pickler(buf, protocol=2).dump({"weight": weight, **extra})
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
            z.writestr("archive/data.pkl", buf.getvalue())
            for key, b in blobs.items():
                z.writestr(f"archive/data/{key}", b)
            z.writestr("archive/version", "3\n")
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    _ = (torch_mod, utils_mod)


def voice_template(out_dir):
    """A tiny RVC-WebUI-layout voice template plus a matching .pth inside a .zip, mirroring
    tools/rvc/make_voice_templates.py: FP16 external initializers named after checkpoint keys,
    a `mobigpt_manifest`, and a speaker table to pad. audio = sine(pitchf) * mean(dec.gain)."""
    import zipfile

    sr, hop = 40000, 400
    manifest, inits, offset = [], [], 0
    shapes = {"dec.gain": [4], "emb_g.weight": [3, 2]}  # template has 3 speaker rows
    nodes = []
    for key, shape in shapes.items():
        nbytes = int(np.prod(shape)) * 2
        t = TensorProto()
        t.name, t.data_type = key + "__fp16", TensorProto.FLOAT16
        t.dims.extend(shape)
        t.data_location = TensorProto.EXTERNAL
        for k, v in (("location", "weights.bin"), ("offset", str(offset)), ("length", str(nbytes))):
            e = t.external_data.add()
            e.key, e.value = k, v
        inits.append(t)
        nodes.append(helper.make_node("Cast", [t.name], [key], to=TensorProto.FLOAT))
        manifest.append(f"{key}\t{offset}\t{','.join(map(str, shape))}")
        offset += (nbytes + 63) // 64 * 64
    inits += [const("two_pi_over_sr", np.array(2 * np.pi / sr, dtype=np.float32)), const("ax2", np.array([2], dtype=np.int64)),
              const("rep", np.array([1, 1, hop], dtype=np.int64)), const("flat", np.array([1, 1, -1], dtype=np.int64)),
              const("tiny", np.array(1e-6, dtype=np.float32))]
    inputs = [helper.make_tensor_value_info("phone", TensorProto.FLOAT, [1, "T", 768]),
              helper.make_tensor_value_info("phone_lengths", TensorProto.INT64, [1]),
              helper.make_tensor_value_info("pitch", TensorProto.INT64, [1, "T"]),
              helper.make_tensor_value_info("pitchf", TensorProto.FLOAT, [1, "T"]),
              helper.make_tensor_value_info("ds", TensorProto.INT64, [1]),
              helper.make_tensor_value_info("rnd", TensorProto.FLOAT, [1, 192, "T"])]
    nodes += [
        helper.make_node("ReduceMean", ["dec.gain"], ["gain"], keepdims=0),
        helper.make_node("ReduceMean", ["emb_g.weight"], ["spk"], keepdims=0),
        helper.make_node("ReduceMean", ["phone"], ["pm"], keepdims=0),
        helper.make_node("ReduceMean", ["rnd"], ["rm"], keepdims=0),
        helper.make_node("Add", ["pm", "rm"], ["prm"]),
        helper.make_node("Add", ["prm", "spk"], ["ties"]),
        helper.make_node("Mul", ["ties", "tiny"], ["ties_s"]),
        helper.make_node("Unsqueeze", ["pitchf", "ax2"], ["f3"]),
        helper.make_node("Tile", ["f3", "rep"], ["f_rep"]),
        helper.make_node("Reshape", ["f_rep", "flat"], ["f_s"]),
        helper.make_node("Mul", ["f_s", "two_pi_over_sr"], ["dphi"]),
        helper.make_node("CumSum", ["dphi", "ax_last"], ["phi"]),
        helper.make_node("Sin", ["phi"], ["s"]),
        helper.make_node("Mul", ["s", "gain"], ["y0"]),
        helper.make_node("Add", ["y0", "ties_s"], ["audio"]),
    ]
    inits.append(const("ax_last", np.array(2, dtype=np.int64)))
    g = helper.make_graph(nodes, "voice_template", inputs, [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, 1, "S"])], inits)
    model = helper.make_model(g)
    meta = {"sample_rate": str(sr), "version": "v2", "f0": "1", "mobigpt_template": "1", "mobigpt_weights_bytes": str(offset),
            "mobigpt_speaker_rows": "3", "mobigpt_manifest": "\n".join(manifest)}
    save(model, os.path.join(out_dir, "rvc_template_v2_40k.onnx"), meta, check=False)

    # The voice: FP32 gain (exercises FP32 -> FP16) and a 2-row speaker table (padded to 3).
    pth = os.path.join(out_dir, "voice_fixture.pth")
    torch_pth(pth, {"dec.gain": np.full(4, 0.25, np.float32), "emb_g.weight": np.ones((2, 2), np.float16),
                    "enc_p.emb_phone.weight": np.zeros((2, 768), np.float16)},
              {"config": [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5]] * 3, [10, 10, 2, 2], 512,
                          [16, 16, 4, 4], 2, 256, sr], "version": "v2", "f0": 1, "sr": "40k", "info": "fixture"})
    with zipfile.ZipFile(os.path.join(out_dir, "voice_fixture.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pth, "My Voice/My Voice.pth")
        z.writestr("My Voice/added_IVF1_Flat_nprobe_1_My_Voice_v2.index", b"\0" * 256)


def kokoro(path, layout="kokoro-onnx"):
    """Kokoro TTS I/O contract. audio = mean(style) / speed, repeated 100x per input token
    (pads included), so tests can check padding, style-row selection and speed exactly."""
    tokens_name, out_name = ("tokens", "audio") if layout == "kokoro-onnx" else ("input_ids", "waveform")
    inputs = [
        helper.make_tensor_value_info(tokens_name, TensorProto.INT64, [1, "T"]),
        helper.make_tensor_value_info("style", TensorProto.FLOAT, [1, 256]),
        helper.make_tensor_value_info("speed", TensorProto.FLOAT, [1]),
    ]
    inits = [
        const("idx1", np.array(1, dtype=np.int64)),
        const("per_token", np.array(100, dtype=np.int64)),
        const("ax0", np.array([0], dtype=np.int64)),
        const("one_shape", np.array([1], dtype=np.int64)),
    ]
    nodes = [
        helper.make_node("Shape", [tokens_name], ["tshape"]),
        helper.make_node("Gather", ["tshape", "idx1"], ["t"]),
        helper.make_node("Mul", ["t", "per_token"], ["n"]),
        helper.make_node("Unsqueeze", ["n", "ax0"], ["n1"]),
        helper.make_node("ReduceMean", ["style"], ["m"], keepdims=0),
        helper.make_node("Reshape", ["m", "one_shape"], ["m1"]),
        helper.make_node("Tile", ["m1", "n1"], ["rep"]),
        helper.make_node("Div", ["rep", "speed"], ["flat"]),
    ]
    out_shape = ["N"]
    if layout == "onnx-community":
        inits.append(const("row_shape", np.array([1, -1], dtype=np.int64)))
        nodes.append(helper.make_node("Reshape", ["flat", "row_shape"], [out_name]))
        out_shape = [1, "N"]
    else:
        nodes.append(helper.make_node("Identity", ["flat"], [out_name]))
    g = helper.make_graph(nodes, "kokoro", inputs, [helper.make_tensor_value_info(out_name, TensorProto.FLOAT, out_shape)], inits)
    save(helper.make_model(g), path)


def kokoro_voice(path, rows=510):
    """Row r is filled with r + 1, so the style chosen for k tokens has mean k."""
    np.repeat(np.arange(1, rows + 1, dtype=np.float32)[:, None], 256, axis=1).astype("<f4").tofile(path)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "fixtures"
    os.makedirs(out, exist_ok=True)
    encoder(os.path.join(out, "enc_webui.onnx"), "webui")
    encoder(os.path.join(out, "enc_hf.onnx"), "hf")
    encoder(os.path.join(out, "enc_wokada.onnx"), "wokada")
    encoder(os.path.join(out, "enc_v1.onnx"), "webui", channels=256)
    rmvpe_mel(os.path.join(out, "rmvpe_mel.onnx"))
    rmvpe_wave(os.path.join(out, "rmvpe_wave.onnx"))
    fcpe(os.path.join(out, "fcpe.onnx"))
    mel_threshold(os.path.join(out, "fcpe_threshold.onnx"))
    mel_threshold(os.path.join(out, "rmvpe_mel_threshold.onnx"), layout="rmvpe")
    synth(os.path.join(out, "synth_webui_40k.onnx"), "webui", 40000)
    synth(
        os.path.join(out, "synth_wokada_48k_fp16.onnx"),
        "wokada",
        48000,
        fp16=True,
        meta={"metadata": json.dumps({"samplingRate": 48000, "f0": True, "embChannels": 768,
                                      "embedder": "hubert_base", "embOutputLayer": 12, "useFinalProj": False})},
    )
    synth(os.path.join(out, "synth_nof0_32k.onnx"), "webui", 32000, with_f0=False, meta={"sample_rate": "32000"})
    synth(os.path.join(out, "synth_v1_40k.onnx"), "webui", 40000, channels=256)
    voice_template(out)
    kokoro(os.path.join(out, "kokoro.onnx"))
    kokoro(os.path.join(out, "kokoro_community.onnx"), "onnx-community")
    kokoro_voice(os.path.join(out, "kokoro_voice.bin"))
    try:
        mel_reference(out)
    except ImportError:
        print("librosa not installed: skipping mel reference data", file=sys.stderr)
    print("fixtures written to", out)


if __name__ == "__main__":
    main()
