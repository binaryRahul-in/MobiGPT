/** Fake voice engine implementing the JSI contract for unit/UI tests. */
import type {EngineInfo, ModelInspection, VoiceEngineConfig, VoiceJsi} from '../../packages/react-native-mobigpt-voice/src/types';

export * from '../../packages/react-native-mobigpt-voice/src/types';

const fs = require('./fs');

export const inspections = new Map<string, ModelInspection>();

const synthInspection = (): ModelInspection => ({
  kind: 'synthesizer',
  sizeMB: 55,
  quantized: false,
  inputs: [],
  outputs: [],
  metadata: {},
  voice: {sampleRate: 40000, channels: 768, usesF0: true, layout: 'rvc-webui', version: 'v2'},
});

let loaded: VoiceEngineConfig | null = null;
export const calls: string[] = [];

export const fakeEngine: VoiceJsi = {
  version: () => ({onnxruntime: '1.24.3', providers: ['XnnpackExecutionProvider', 'CPUExecutionProvider'], engine: 'mobigpt-rvc/1'}),
  load: jest.fn(async (cfg: VoiceEngineConfig): Promise<EngineInfo> => {
    calls.push('load');
    loaded = cfg;
    return {
      sampleRate: 40000,
      channels: 768,
      usesF0: true,
      layout: 'rvc-webui',
      version: 'v2',
      pitchMethod: cfg.pitchMethod,
      providers: ['synth:xnnpack', 'encoder:xnnpack'],
      warnings: [],
      indexUsed: false,
    };
  }),
  unload: jest.fn(async () => {
    loaded = null;
  }),
  isLoaded: () => loaded != null,
  setPitchShift: jest.fn(),
  convertFile: jest.fn(async (_in: string, out: string, onProgress?: (p: number) => void) => {
    onProgress?.(0.5);
    onProgress?.(1);
    fs.__putFile(out, 96000);
    return {
      inputSeconds: 3,
      outputSeconds: 3,
      wallMs: 900,
      realtimeFactor: 0.3,
      chunks: 2,
      sampleRate: 40000,
      cancelled: false,
      stages: {encoderMs: 300, pitchMs: 200, synthMs: 400, totalMs: 900},
    };
  }),
  cancel: jest.fn(),
  startLive: jest.fn(async () => undefined),
  stopLive: jest.fn(async () => undefined),
  liveStats: () => ({running: true, latencyMs: 2900, lastChunkMs: 400, realtimeFactor: 0.16, chunks: 3, droppedBlocks: 0, inputLevel: 0.2, outputLevel: 0.3, error: ''}),
  startRecording: jest.fn(async () => undefined),
  stopRecording: jest.fn(async () => {
    fs.__putFile('/docs/mobigpt/voice/recordings/rec.wav', 32000);
    return {path: '/docs/mobigpt/voice/recordings/rec.wav', seconds: 1, sampleRate: 16000, peak: 0.5};
  }),
  recordingLevel: () => 0.4,
  play: jest.fn(async () => ({completed: true})),
  stopPlayback: jest.fn(),
  isPlaying: () => false,
  inspect: jest.fn(async (path: string) => inspections.get(path) ?? synthInspection()),
  analyzePitch: jest.fn(async () => [0, 220, 221]),
  benchmark: jest.fn(async (seconds: number) => ({
    audioSeconds: seconds,
    realtimeFactor: 0.25,
    sampleRate: 40000,
    stages: {encoderMs: 400, pitchMs: 300, synthMs: 800, totalMs: 1500},
    providers: ['synth:xnnpack'],
  })),
};

export const isVoiceModuleAvailable = () => true;
export const getVoiceEngine = jest.fn(async () => fakeEngine);
export const VoiceNative = {
  install: jest.fn(async () => true),
  synthesizeSpeech: jest.fn(async (_t: string, out: string) => {
    fs.__putFile(out, 44100);
    return {path: out, sampleRate: 22050, seconds: 1};
  }),
  listTtsVoices: jest.fn(async () => [
    {id: 'en-us-x-1', name: 'English', language: 'en-US', quality: 400, requiresNetwork: false},
    {id: 'net', name: 'Network', language: 'en-US', quality: 500, requiresNetwork: true},
  ]),
  decodeAudioToWav: jest.fn(async (_i: string, out: string) => {
    fs.__putFile(out, 64000);
    return {path: out, sampleRate: 44100, seconds: 2};
  }),
};
