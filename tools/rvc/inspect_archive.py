#!/usr/bin/env python3
"""Describe an RVC voice archive (.zip with .pth/.index, or a bare .pth) without PyTorch.

Prints what MobiGPT needs to know to run it: RVC version (v1 256-d / v2 768-d),
sample rate, pitch guidance (f0), speaker count, weight dtype, weight-norm layout
and whether any tensor keys fall outside the standard SynthesizerTrnMs*NSFsid layout.

Usage: python tools/rvc/inspect_archive.py VOICE.zip|VOICE.pth [--json out.json]
"""
import argparse
import io
import json
import pickle
import sys
import zipfile


class _Tensor:
    def __init__(self, storage, offset, size, stride):
        self.dtype, self.key = storage
        self.offset, self.shape, self.stride = offset, tuple(size), tuple(stride)


class _Unpickler(pickle.Unpickler):
    """Reads torch.save() pickles with tensors left as (dtype, storage key, shape) stubs."""

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return lambda storage, offset, size, stride, *rest: _Tensor(storage, offset, size, stride)
        if module == "torch" and name.endswith("Storage"):
            return name  # e.g. "HalfStorage"
        if module == "collections" and name == "OrderedDict":
            import collections

            return collections.OrderedDict
        if module == "torch" and name in ("float16", "float32", "bfloat16"):
            return name
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")

    def persistent_load(self, pid):
        # ('storage', storage_type, key, location, numel)
        _, storage_type, key, _, _ = pid
        dtype = storage_type if isinstance(storage_type, str) else str(storage_type)
        return dtype.replace("Storage", "").lower(), key


def load_pth(data: bytes) -> dict:
    z = zipfile.ZipFile(io.BytesIO(data))
    pkl = next(n for n in z.namelist() if n.endswith("data.pkl"))
    return _Unpickler(io.BytesIO(z.read(pkl))).load()


def describe(ckpt: dict) -> dict:
    weight = ckpt.get("weight", ckpt.get("model", {}))
    cfg = ckpt.get("config")
    keys = list(weight.keys()) if isinstance(weight, dict) else []
    dtypes = sorted({t.dtype for t in weight.values() if isinstance(t, _Tensor)})
    emb_phone = weight.get("enc_p.emb_phone.weight")
    emb_g = weight.get("emb_g.weight")
    return {
        "version": ckpt.get("version", "v1"),
        "f0": int(ckpt.get("f0", 1)),
        "sr_tag": ckpt.get("sr"),
        "sample_rate": cfg[-1] if isinstance(cfg, (list, tuple)) else None,
        "config": cfg,
        "feature_dim": emb_phone.shape[1] if emb_phone else None,
        "speakers": emb_g.shape[0] if emb_g else None,
        "tensors": len(keys),
        "dtypes": dtypes,
        "weight_norm": any(k.endswith("weight_g") for k in keys),
        "has_enc_q": any(k.startswith("enc_q.") for k in keys),
        "info": ckpt.get("info"),
        "top_level_keys": sorted(ckpt.keys()),
        "unusual_keys": [k for k in keys if not k.split(".")[0] in ("enc_p", "dec", "flow", "emb_g", "enc_q")][:20],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--json")
    args = ap.parse_args()
    raw = open(args.path, "rb").read()
    report = {"archive": args.path, "entries": [], "voices": []}
    outer = zipfile.ZipFile(io.BytesIO(raw))
    is_torch = any(n.endswith("data.pkl") for n in outer.namelist())
    if is_torch:
        report["voices"].append({"file": args.path, **describe(load_pth(raw))})
    else:
        for info in outer.infolist():
            report["entries"].append({"name": info.filename, "bytes": info.file_size, "compressed": info.compress_type != 0})
            if info.filename.lower().endswith(".pth"):
                try:
                    report["voices"].append({"file": info.filename, **describe(load_pth(outer.read(info)))})
                except Exception as e:  # noqa: BLE001
                    report["voices"].append({"file": info.filename, "error": str(e)})
    print(json.dumps(report, indent=2, default=str))
    if args.json:
        json.dump(report, open(args.json, "w"), indent=2, default=str)
    return 0 if report["voices"] else 1


if __name__ == "__main__":
    sys.exit(main())
