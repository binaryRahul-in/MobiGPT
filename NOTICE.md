# Third-party notices

MobiGPT is an independent project with its own name, branding and code base.
It builds on, and in a few places adapts code from, the following open-source
projects. Their licenses are reproduced or linked below.

| Component | Use in MobiGPT | License |
|---|---|---|
| [PocketPal AI](https://github.com/a-ghorbani/pocketpal-ai) © 2024 Asghar Ghorbani | Architecture reference for the on-device LLM app; adapted: hardware-info native module (`packages/react-native-mobigpt-device`), GGUF memory estimator (`src/utils/gguf.ts`), OpenCL/Hexagon capability rules | MIT |
| [llama.rn](https://github.com/mybigday/llama.rn) / [llama.cpp](https://github.com/ggml-org/llama.cpp) | LLM inference (CPU, OpenCL, Metal, Hexagon) | MIT |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | Voice engine inference (CPU, XNNPACK, NNAPI, QNN, Core ML) | MIT |
| [WORLD](https://github.com/mmorise/World) © 2010 M. Morise | DIO, Harvest and StoneMask pitch trackers, vendored in `packages/react-native-mobigpt-voice/cpp/third_party/world` | BSD-3-Clause |
| [RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | Reference pipeline (feature 2x upsampling, coarse pitch, RMVPE decoding); its ONNX model definition is exported by `tools/rvc/export_voice_onnx.py` and into the weight-free voice templates used for on-device `.pth` import | MIT |
| [FCPE / torchfcpe](https://github.com/CNChTu/FCPE) | Reference mel front-end + local-argmax decoder | MIT |
| [voiceclonnx](https://github.com/TigreGotico/voiceclonnx) | Reference ONNX exports and INT8 measurements for ContentVec/RMVPE | MIT |
| [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) © hexgrad | Neural TTS model, downloaded at runtime; its phoneme vocabulary is embedded in `src/services/tts/vocab.ts` | Apache-2.0 |
| [misaki](https://github.com/hexgrad/misaki) © hexgrad | English G2P: lexicons downloaded at runtime; lookup and `-s/-ed/-ing` rules ported to `src/services/tts/g2p.ts`; sample entries in `jest/fixtures-lexicon.json` | Apache-2.0 |
| [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) | INT8 ONNX export of Kokoro and the reference inference loop (padding, style row per token count) | MIT |
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) / Moonshine | Speech recogniser used only in CI to check TTS intelligibility | Apache-2.0 / MIT |
| [RVC-Android](https://github.com/ouor/RVC-Android), [w-okada voice-changer](https://github.com/w-okada/voice-changer) | Reference for the w-okada ONNX I/O layout (feats/p_len/sid, metadata JSON) | MIT |

Model weights downloaded at runtime are governed by their own licenses, shown in-app next to each model.

## WORLD license (BSD-3-Clause)

See `packages/react-native-mobigpt-voice/cpp/third_party/world/LICENSE.txt`.

## PocketPal AI license (MIT)

Copyright (c) 2024 Asghar Ghorbani — permission is hereby granted, free of charge, to any person obtaining a copy of this software… (full text: https://github.com/a-ghorbani/pocketpal-ai/blob/main/LICENSE).
