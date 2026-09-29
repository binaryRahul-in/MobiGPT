#!/usr/bin/env node
// Prints the effective build-time feature configuration.
const features = require('../mobigpt.features.json');
const voice = process.env.MOBIGPT_VOICE ? process.env.MOBIGPT_VOICE !== '0' : features.voice?.enabled !== false;
console.log('MobiGPT build features');
console.log(`  llm   : ${features.llm?.enabled !== false ? 'on' : 'off'} (llama.cpp via llama.rn)`);
console.log(`  voice : ${voice ? 'on' : 'off'} (ONNX Runtime ${features.voice?.ortVersion}, flavour ${features.voice?.ortFlavor})`);
