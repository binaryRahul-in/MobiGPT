#!/usr/bin/env python3
"""Builds the weight-free RVC graph templates that let the app import .pth voices on the phone.

A template is RVC's ONNX graph (models_onnx.SynthesizerTrnMsNSFsidM) for one
architecture (v1/v2 x 32/40/48 kHz) whose every weight lives in an external file:

    voice dir/model.onnx    the template (graph only, ~1 MB)
    voice dir/weights.bin   the voice's tensors as FP16, at the offsets listed in the
                            template's `mobigpt_manifest` metadata (key, offset, dims per
                            line; written on the device by cpp/rvc/Checkpoint.cpp)

Every initializer is named after its key in the RVC checkpoint (exported from unique
random values so the exporter cannot merge equal tensors), stored as FLOAT16 and cast
to FP32 inside the graph, which ONNX Runtime folds away at load time. Importing a
voice is then a byte copy from the .pth into weights.bin - no PyTorch on the phone.

Usage:
  python tools/rvc/make_voice_templates.py --rvc-repo RVC --out templates [--check VOICE.pth]
"""
import argparse
import json
import os
import sys

import numpy as np

# (version, sample rate) -> RVC config file. RVC's v2 40 kHz voices use the v1 40k config.
VARIANTS = {
    ("v1", 32000): "configs/v1/32k.json",
    ("v1", 40000): "configs/v1/40k.json",
    ("v1", 48000): "configs/v1/48k.json",
    ("v2", 32000): "configs/v2/32k.json",
    ("v2", 40000): "configs/v1/40k.json",
    ("v2", 48000): "configs/v2/48k.json",
}
SPEAKER_ROWS = 109  # RVC's default spk_embed_dim; other counts are padded/truncated on import
WEIGHTS_FILE = "weights.bin"


def rvc_config(repo: str, cfg_file: str, sr: int) -> list:
    d = json.load(open(os.path.join(repo, cfg_file)))
    m = d["model"]
    return [d["data"]["filter_length"] // 2 + 1, d["train"]["segment_size"] // d["data"]["hop_length"],
            m["inter_channels"], m["hidden_channels"], m["filter_channels"], m["n_heads"], m["n_layers"],
            m["kernel_size"], m["p_dropout"], m["resblock"], m["resblock_kernel_sizes"], m["resblock_dilation_sizes"],
            m["upsample_rates"], m["upsample_initial_channel"], m["upsample_kernel_sizes"], SPEAKER_ROWS,
            m["gin_channels"], sr]


def export_template(repo: str, version: str, sr: int, out_path: str) -> dict:
    import inspect

    import onnx
    import torch

    sys.path.insert(0, os.path.abspath(repo))
    from infer.lib.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM  # type: ignore

    cfg = rvc_config(repo, VARIANTS[(version, sr)], sr)
    net = SynthesizerTrnMsNSFsidM(*cfg, is_half=False, version=version).eval()
    torch.manual_seed(1)
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(torch.randn_like(p))  # unique values: the exporter must not merge any two tensors
    channels = 768 if version == "v2" else 256
    T = 200
    dummy = (torch.rand(1, T, channels), torch.tensor([T]).long(), torch.randint(size=(1, T), low=5, high=255),
             torch.rand(1, T) * 300, torch.LongTensor([0]), torch.rand(1, 192, T))
    raw = out_path + ".raw.onnx"
    torch.onnx.export(net, dummy, raw, input_names=["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"],
                      output_names=["audio"], dynamic_axes={"phone": [1], "pitch": [1], "pitchf": [1], "rnd": [2], "audio": [2]},
                      do_constant_folding=False, opset_version=17,
                      **({"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}))
    model = onnx.load(raw)
    os.remove(raw)
    state = {k for k in net.state_dict().keys()}

    manifest, casts, offset = [], [], 0
    for init in model.graph.initializer:
        if init.name not in state:
            raise RuntimeError(f"initializer {init.name} is not a checkpoint tensor")
        if init.data_type != onnx.TensorProto.FLOAT:
            raise RuntimeError(f"{init.name}: expected float32")
        dims = list(init.dims)
        numel = int(np.prod(dims)) if dims else 1
        nbytes = numel * 2
        key = init.name
        manifest.append({"key": key, "shape": dims, "offset": offset})
        init.name = key + "__fp16"
        init.data_type = onnx.TensorProto.FLOAT16
        init.ClearField("raw_data")
        init.ClearField("float_data")
        init.data_location = onnx.TensorProto.EXTERNAL
        del init.external_data[:]
        for k, v in (("location", WEIGHTS_FILE), ("offset", str(offset)), ("length", str(nbytes))):
            e = init.external_data.add()
            e.key, e.value = k, v
        casts.append(onnx.helper.make_node("Cast", [init.name], [key], to=onnx.TensorProto.FLOAT, name=key + "__to_fp32"))
        offset += (nbytes + 63) // 64 * 64  # 64-byte aligned tensors
    nodes = casts + list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    meta = {"sample_rate": str(sr), "version": version, "f0": "1", "source": "mobigpt voice template",
            "mobigpt_template": "1", "mobigpt_weights_bytes": str(offset), "mobigpt_speaker_rows": str(SPEAKER_ROWS),
            # One line per tensor: key \t byte offset \t comma-separated dims (parsed by the C++ importer).
            "mobigpt_manifest": "\n".join(f"{e['key']}\t{e['offset']}\t{','.join(map(str, e['shape']))}" for e in manifest)}
    for k, v in meta.items():
        p = model.metadata_props.add()
        p.key, p.value = k, v
    onnx.save(model, out_path)
    return {"version": version, "sample_rate": sr, "tensors": len(manifest), "weights_bytes": offset,
            "file": os.path.basename(out_path), "bytes": os.path.getsize(out_path)}


def fill_from_pth(template: str, pth: str, out_dir: str) -> str:
    """Python reference for the app's native importer: template + .pth -> voice directory."""
    import shutil

    import onnx
    import torch

    meta = {p.key: p.value for p in onnx.load(template, load_external_data=False).metadata_props}
    manifest = [{"key": k, "offset": int(o), "shape": [int(d) for d in dims.split(",") if d]}
                for k, o, dims in (line.split("\t") for line in meta["mobigpt_manifest"].splitlines())]
    weights = torch.load(pth, map_location="cpu", weights_only=False)["weight"]
    os.makedirs(out_dir, exist_ok=True)
    buf = bytearray(int(meta["mobigpt_weights_bytes"]))
    rows = int(meta["mobigpt_speaker_rows"])
    for e in manifest:
        t = weights[e["key"]].detach().to(torch.float16).contiguous().numpy()
        if e["key"] == "emb_g.weight" and t.shape[0] != rows:  # pad/truncate the speaker table
            fixed = np.zeros((rows, t.shape[1]), np.float16)
            fixed[: min(rows, t.shape[0])] = t[:rows]
            t = fixed
        assert list(t.shape) == e["shape"], (e["key"], t.shape, e["shape"])
        b = t.tobytes()
        buf[e["offset"]: e["offset"] + len(b)] = b
    open(os.path.join(out_dir, WEIGHTS_FILE), "wb").write(bytes(buf))
    shutil.copy(template, os.path.join(out_dir, "model.onnx"))
    return os.path.join(out_dir, "model.onnx")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rvc-repo", required=True)
    ap.add_argument("--out", default="voice-templates")
    ap.add_argument("--only", help="e.g. v2-40000")
    ap.add_argument("--check", help="a .pth to import through the template and compare with export_voice_onnx.py")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    built = []
    for (version, sr) in VARIANTS:
        if args.only and args.only != f"{version}-{sr}":
            continue
        path = os.path.join(args.out, f"rvc_template_{version}_{sr // 1000}k.onnx")
        info = export_template(args.rvc_repo, version, sr, path)
        built.append(info)
        print(json.dumps(info))
    json.dump(built, open(os.path.join(args.out, "templates.json"), "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
