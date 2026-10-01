export type PitchMethod = 'rmvpe' | 'fcpe' | 'harvest' | 'dio' | 'pm';
export type Accelerator = 'auto' | 'cpu' | 'xnnpack' | 'nnapi' | 'qnn' | 'coreml';
export type LoadStrategy = 'resident' | 'sequential';

export interface VoiceEngineConfig {
  /** HuBERT / ContentVec ONNX (fp32 or INT8). */
  encoderPath: string;
  /** Per-voice RVC synthesiser (net_g) ONNX. */
  voicePath: string;
  pitchMethod: PitchMethod;
  /** Required for 'rmvpe' and 'fcpe'; ignored by DSP trackers. */
  pitchModelPath?: string;
  accelerator?: Accelerator;
  threads?: number;
  lowMemory?: boolean;
  fp16?: boolean;
  loadStrategy?: LoadStrategy;
  pitchShift?: number;
  speakerId?: number;
  rmsMixRate?: number;
  /** Accepted for desktop-preset compatibility; always forced to 0 on device. */
  indexRate?: number;
  chunkSeconds?: number;
  contextSeconds?: number;
  crossfadeMs?: number;
  solaSearchMs?: number;
  f0Min?: number;
  f0Max?: number;
  defaultSampleRate?: number;
}

export interface EngineInfo {
  sampleRate: number;
  channels: number;
  usesF0: boolean;
  layout: 'rvc-webui' | 'w-okada' | string;
  version: string;
  pitchMethod: PitchMethod;
  providers: string[];
  warnings: string[];
  indexUsed: boolean;
}

export interface StageTimings {
  encoderMs: number;
  pitchMs: number;
  synthMs: number;
  totalMs: number;
}

export interface ConversionStats {
  inputSeconds: number;
  outputSeconds: number;
  wallMs: number;
  realtimeFactor: number;
  chunks: number;
  sampleRate: number;
  cancelled: boolean;
  stages: StageTimings;
}

export interface LiveStats {
  running: boolean;
  latencyMs: number;
  lastChunkMs: number;
  realtimeFactor: number;
  chunks: number;
  droppedBlocks: number;
  inputLevel: number;
  outputLevel: number;
  error: string;
}

export interface RecordingResult {
  path: string;
  seconds: number;
  sampleRate: number;
  peak: number;
}

export interface TensorDescription {
  name: string;
  type: string;
  shape: number[];
}

export interface ModelInspection {
  kind: 'synthesizer' | 'encoder' | 'rmvpe' | 'fcpe' | 'unknown';
  sizeMB: number;
  quantized: boolean;
  inputs: TensorDescription[];
  outputs: TensorDescription[];
  metadata: Record<string, string>;
  voice?: {
    sampleRate: number;
    channels: number;
    usesF0: boolean;
    layout: string;
    version: 'v1' | 'v2';
  };
}

export interface VoiceBenchmark {
  audioSeconds: number;
  realtimeFactor: number;
  sampleRate: number;
  stages: StageTimings;
  providers: string[];
}

/** Shape of `global.__MobiGPTVoice` installed by the native module. */
export interface VoiceJsi {
  version(): {onnxruntime: string; providers: string[]; engine: string};
  load(config: VoiceEngineConfig): Promise<EngineInfo>;
  unload(): Promise<void>;
  isLoaded(): boolean;
  setPitchShift(semitones: number): void;
  convertFile(inputWav: string, outputWav: string, onProgress?: (fraction: number) => void): Promise<ConversionStats>;
  cancel(): void;
  startLive(): Promise<void>;
  stopLive(): Promise<void>;
  liveStats(): LiveStats;
  startRecording(path: string): Promise<void>;
  stopRecording(): Promise<RecordingResult>;
  recordingLevel(): number;
  play(path: string): Promise<{completed: boolean}>;
  stopPlayback(): void;
  isPlaying(): boolean;
  inspect(path: string): Promise<ModelInspection>;
  analyzePitch(path: string, method: PitchMethod, modelPath?: string): Promise<number[]>;
  benchmark(seconds: number): Promise<VoiceBenchmark>;

  // RVC voices shared as PyTorch checkpoints (.pth, or a .zip holding one).
  rvcInfo(path: string): Promise<RvcCheckpointInfo>;
  /** Writes outDir/model.onnx (a copy of the template) + outDir/weights.bin. */
  rvcImport(
    path: string,
    templatePath: string,
    outDir: string,
  ): Promise<{modelPath: string; weightsBytes: number; info: RvcCheckpointInfo}>;

  // Neural text-to-speech (Kokoro-82M). Independent of the RVC engine.
  ttsLoad(modelPath: string, options?: {accelerator?: Accelerator; threads?: number; lowMemory?: boolean}): Promise<TtsModelInfo>;
  ttsUnload(): Promise<void>;
  ttsIsLoaded(): boolean;
  ttsSynthesize(request: TtsRequest): Promise<TtsResult>;
}

export interface RvcCheckpointInfo {
  pthName: string;
  version: 'v1' | 'v2' | string;
  sampleRate: number;
  f0: boolean;
  speakers: number;
  featureDim: number;
  dtype: string;
  tensors: number;
  /** A FAISS .index came with the voice; MobiGPT does not need it (index_rate = 0). */
  hasIndex: boolean;
  info: string;
  /** Graph template file needed to import it, e.g. "rvc_template_v2_40k.onnx". */
  template: string;
}

export interface TtsModelInfo {
  provider: string;
  warnings: string[];
}

export interface TtsRequest {
  /** Phoneme-token windows (<= 510 ids each), from the app's G2P. */
  windows: number[][];
  /** Seconds of silence after each window. */
  pauses: number[];
  /** Kokoro voice style file (N x 256 float32). */
  voicePath: string;
  speed: number;
  outputPath: string;
}

export interface TtsResult {
  path: string;
  seconds: number;
  sampleRate: number;
  inferMs: number;
  realtimeFactor: number;
  windows: number;
}
