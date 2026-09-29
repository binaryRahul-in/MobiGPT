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
| [RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | Reference pipeline (feature 2x upsampling, coarse pitch, RMVPE decoding, ONNX export in `tools/rvc/export_voice_onnx.py`) | MIT |
| [FCPE / torchfcpe](https://github.com/CNChTu/FCPE) | Reference mel front-end + local-argmax decoder | MIT |
| [voiceclonnx](https://github.com/TigreGotico/voiceclonnx) | Reference ONNX exports and INT8 measurements for ContentVec/RMVPE | MIT |
| [RVC-Android](https://github.com/ouor/RVC-Android), [w-okada voice-changer](https://github.com/w-okada/voice-changer) | Reference for the w-okada ONNX I/O layout (feats/p_len/sid, metadata JSON) | MIT |

Model weights downloaded at runtime are governed by their own licenses, shown in-app next to each model.

## WORLD license (BSD-3-Clause)

See `packages/react-native-mobigpt-voice/cpp/third_party/world/LICENSE.txt`.

## PocketPal AI license (MIT)

Copyright (c) 2024 Asghar Ghorbani — permission is hereby granted, free of charge, to any person obtaining a copy of this software… (full text: https://github.com/a-ghorbani/pocketpal-ai/blob/main/LICENSE).
