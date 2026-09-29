#!/usr/bin/env python3
"""
Convert a community RVC voice (.pth) into the ONNX synthesiser MobiGPT loads.

Uses RVC-Project's own ONNX-friendly model definition (MIT) so the graph and
I/O names match what "Export ONNX" in RVC WebUI produces:
    inputs  phone[1,T,256|768] phone_lengths[1] pitch[1,T] pitchf[1,T] ds[1] rnd[1,192,T]
    output  audio[1,1,T*hop]
and embeds metadata (sample_rate, version, f0) so the app can auto-configure.

    git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI rvc
    pip install torch onnx onnxconverter-common
    python export_voice_onnx.py --rvc-repo rvc my_voice.pth my_voice.onnx [--fp16]

The FAISS .index that often ships next to a .pth is intentionally ignored:
MobiGPT runs with index_rate = 0 on mobile.
"""
import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rvc-repo", required=True, help="path to a clone of RVC-Project/Retrieval-based-Voice-Conversion-WebUI")
    ap.add_argument("--fp16", action="store_true", help="store weights in fp16 (keeps fp32 I/O); ~2x smaller")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("pth")
    ap.add_argument("onnx_out")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.rvc_repo))
    import onnx
    import torch
    from infer.lib.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM  # type: ignore

    cpt = torch.load(args.pth, map_location="cpu", weights_only=False)
    version = cpt.get("version", "v1")
    if_f0 = int(cpt.get("f0", 1))
    sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # number of speakers
    channels = 256 if version == "v1" else 768
    if not if_f0:
        print("warning: this is a no-f0 voice; MobiGPT supports it but pitch shifting has no effect", file=sys.stderr)

    net_g = SynthesizerTrnMsNSFsidM(*cpt["config"], is_half=False, version=version)
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g.eval()

    T = 200
    dummy = (
        torch.rand(1, T, channels),
        torch.tensor([T]).long(),
        torch.randint(size=(1, T), low=5, high=255),
        torch.rand(1, T) * 300,
        torch.LongTensor([0]),
        torch.rand(1, 192, T),
    )
    torch.onnx.export(
        net_g,
        dummy,
        args.onnx_out,
        input_names=["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"],
        output_names=["audio"],
        dynamic_axes={"phone": [1], "pitch": [1], "pitchf": [1], "rnd": [2], "audio": [2]},
        do_constant_folding=True,
        opset_version=args.opset,
    )

    model = onnx.load(args.onnx_out)
    if args.fp16:
        from onnxconverter_common import float16

        model = float16.convert_float_to_float16(model, keep_io_types=True)
    for k, v in {"sample_rate": str(sr), "version": version, "f0": str(if_f0), "source": "mobigpt export_voice_onnx"}.items():
        p = model.metadata_props.add()
        p.key, p.value = k, v
    onnx.save(model, args.onnx_out)
    print(f"wrote {args.onnx_out}: RVC {version}, {sr} Hz, f0={bool(if_f0)}, {os.path.getsize(args.onnx_out) / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
