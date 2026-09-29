import {DeviceProfile, gpuOffloadSupport, npuSupport} from './device';
import {Issue, Requirement} from './requirements';

/**
 * Every optional capability in MobiGPT is a *feature* with explicit hardware
 * requirements, optional downloadable packs and configuration. Nothing heavy
 * is installed until the user opts in, and each install shows the exact
 * warnings for the device in hand.
 */

export type FeatureId = 'chat' | 'gpu' | 'npu' | 'voice' | 'liveVoice' | 'ttsVoice' | 'benchmarks';

export interface FeatureOptionChoice<T extends string | number | boolean> {
  value: T;
  label: string;
  description?: string;
  /** Extra requirements that apply only when this choice is selected. */
  requirement?: Requirement;
}

export interface FeatureOption {
  key: string;
  label: string;
  description: string;
  type: 'choice' | 'toggle' | 'slider' | 'locked';
  choices?: FeatureOptionChoice<string>[];
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  /** Locked options are fixed by design and only explained to the user. */
  lockedReason?: string;
}

export interface FeatureDefinition {
  id: FeatureId;
  title: string;
  summary: string;
  icon: string;
  core?: boolean;
  experimental?: boolean;
  dependsOn?: FeatureId[];
  requirement: Requirement;
  options?: FeatureOption[];
  /** Human readable download footprint for the default pack. */
  footprint?: string;
}

const MB = 1e6;

export const FEATURES: FeatureDefinition[] = [
  {
    id: 'chat',
    title: 'On-device chat (LLM)',
    summary: 'Run GGUF language models locally with llama.cpp. Private, offline, no account.',
    icon: 'chat-processing-outline',
    core: true,
    requirement: {minRamGB: 2, recommendedRamGB: 4, minCores: 4, cpuFeaturesRecommended: ['dotprod'], warnOnEmulator: true},
    footprint: 'Models from 145 MB',
  },
  {
    id: 'gpu',
    title: 'GPU acceleration',
    summary: 'Offload transformer layers to the GPU (OpenCL on Adreno, Metal on Apple silicon).',
    icon: 'expansion-card',
    dependsOn: ['chat'],
    requirement: {
      custom: (p: DeviceProfile): Issue | null => {
        const g = gpuOffloadSupport(p);
        return g.supported ? null : {severity: 'block', code: 'gpu', message: g.reason ?? 'GPU offload unsupported'};
      },
    },
  },
  {
    id: 'npu',
    title: 'NPU acceleration (Hexagon)',
    summary: "Run supported layers on the Qualcomm Hexagon NPU via llama.cpp's HTP backend.",
    icon: 'chip',
    experimental: true,
    dependsOn: ['chat'],
    requirement: {
      platforms: ['android'],
      custom: (p: DeviceProfile): Issue | null => {
        const n = npuSupport(p);
        if (!n.supported) {
          return {severity: 'block', code: 'npu', message: n.reason ?? 'NPU unsupported'};
        }
        return {
          severity: 'warn',
          code: 'npu-experimental',
          message: n.reason ?? `Hexagon ${n.generation} detected. NPU offload is experimental: some ops fall back to CPU.`,
        };
      },
    },
  },
  {
    id: 'voice',
    title: 'Voice Studio (RVC)',
    summary:
      'Speech-to-speech voice conversion with Retrieval-based Voice Conversion on ONNX Runtime. Import your own RVC voices or pick community voices.',
    icon: 'account-voice',
    dependsOn: [],
    requirement: {
      nativeModule: 'voice',
      minRamGB: 2.5,
      recommendedRamGB: 6,
      minCores: 4,
      recommendedCores: 6,
      storageBytes: 200 * MB,
      cpuFeaturesRecommended: ['dotprod', 'fp16'],
    },
    footprint: '≈ 200 MB (INT8 encoder + FCPE) · up to 740 MB (FP32 + RMVPE)',
    options: [
      {
        key: 'pitchMethod',
        label: 'Pitch extractor',
        description: 'How F0 (the melody of your voice) is tracked. Neural trackers are most robust; DSP trackers need zero extra RAM.',
        type: 'choice',
        choices: [
          {
            value: 'rmvpe',
            label: 'RMVPE (neural, best quality)',
            description: 'U-Net pitch model. INT8 ≈ 99 MB, FP32 ≈ 362 MB.',
            requirement: {recommendedRamGB: 6},
          },
          {
            value: 'fcpe',
            label: 'FCPE (neural, lightweight)',
            description: '≈ 42 MB ONNX tracker, ~5-10x faster than RMVPE with similar accuracy on clean speech.',
          },
          {
            value: 'harvest',
            label: 'Harvest (DSP, zero RAM)',
            description: 'WORLD Harvest in native C++. Accurate but the slowest option (~0.3-0.6x RT on mid-range CPUs).',
          },
          {
            value: 'dio',
            label: 'DIO + StoneMask (DSP, zero RAM)',
            description: 'WORLD DIO with StoneMask refinement. Fast and light; slightly less robust in noise.',
          },
          {
            value: 'pm',
            label: 'PM (DSP, fastest)',
            description: 'Normalised autocorrelation tracker. Near-zero cost; best for clean, close-mic speech.',
          },
        ],
      },
      {
        key: 'encoderPrecision',
        label: 'Content encoder precision',
        description: 'INT8 dynamic quantisation shrinks ContentVec/HuBERT 4x and speeds it up on ARM; FP32 keeps maximum articulation.',
        type: 'choice',
        choices: [
          {value: 'int8', label: 'INT8 (recommended)', description: '≈ 95 MB on disk, ≈ 180 MB while loaded.'},
          {
            value: 'fp32',
            label: 'FP32 (reference)',
            description: '≈ 378 MB on disk, ≈ 600 MB while loaded.',
            requirement: {recommendedRamGB: 8},
          },
        ],
      },
      {
        key: 'indexRate',
        label: 'Retrieval index (FAISS)',
        description: 'index_rate is hard-wired to 0: the .index vector database is never loaded.',
        type: 'locked',
        lockedReason: 'Saves 100–500 MB of RAM per voice and removes a ~1 s load. Timbre comes entirely from the voice model.',
      },
      {
        key: 'chunkSeconds',
        label: 'Streaming chunk length',
        description: 'Audio is converted in fixed windows (with context + SOLA crossfade) so memory stays flat for any input length.',
        type: 'slider',
        min: 2,
        max: 3,
        step: 0.25,
        unit: 's',
      },
      {
        key: 'nativePath',
        label: 'Native tensor path',
        description: 'HuBERT, pitch tracking and net_g run entirely in C++ (JSI); audio streams straight to AudioTrack / AVAudioEngine.',
        type: 'locked',
        lockedReason: 'No tensor ever crosses into JavaScript, so there is no serialisation overhead or GC pressure.',
      },
      {
        key: 'accelerator',
        label: 'Accelerator',
        description: 'ONNX Runtime execution provider. Unsupported choices fall back to CPU automatically.',
        type: 'choice',
        choices: [
          {value: 'auto', label: 'Auto', description: 'XNNPACK on Android, CPU (MLAS) on iOS.'},
          {value: 'cpu', label: 'CPU (MLAS)'},
          {value: 'xnnpack', label: 'XNNPACK', description: 'Optimised ARM/x86 kernels.'},
          {
            value: 'nnapi',
            label: 'NNAPI (GPU/DSP)',
            description: 'Android only; deprecated in Android 15 and often slower for dynamic shapes.',
            requirement: {platforms: ['android']},
          },
          {
            value: 'qnn',
            label: 'QNN (Hexagon NPU)',
            description: 'Requires the QNN build flavour (voice.ortFlavor = "qnn").',
            requirement: {platforms: ['android']},
          },
          {value: 'coreml', label: 'Core ML (Neural Engine)', description: 'iOS only.', requirement: {platforms: ['ios']}},
        ],
      },
      {
        key: 'loadStrategy',
        label: 'Model loading',
        description:
          'Resident keeps all three models in RAM (fastest, needed for live mode). Sequential loads one model per stage for the lowest peak memory.',
        type: 'choice',
        choices: [
          {value: 'resident', label: 'Resident (fast)'},
          {value: 'sequential', label: 'Sequential (low memory)'},
        ],
      },
    ],
  },
  {
    id: 'liveVoice',
    title: 'Live voice changer',
    summary: 'Microphone → RVC → speaker in real time, fully native. Latency ≈ chunk length.',
    icon: 'microphone-message',
    experimental: true,
    dependsOn: ['voice'],
    requirement: {
      nativeModule: 'voice',
      minRamGB: 4,
      recommendedRamGB: 8,
      minCores: 6,
      recommendedCores: 8,
      recommendedMaxFreqMhz: 2800,
      warnOnEmulator: true,
    },
  },
  {
    id: 'ttsVoice',
    title: 'Text → speech → voice',
    summary: 'Type text, synthesise it with the offline system TTS, then convert it into any RVC voice.',
    icon: 'text-to-speech',
    dependsOn: ['voice'],
    requirement: {nativeModule: 'voice', minRamGB: 2.5, recommendedRamGB: 4},
  },
  {
    id: 'benchmarks',
    title: 'Benchmarks',
    summary: 'Measure prompt processing / generation speed and voice real-time factor on this device.',
    icon: 'speedometer',
    core: true,
    requirement: {},
  },
];

export function getFeature(id: FeatureId): FeatureDefinition {
  const f = FEATURES.find(x => x.id === id);
  if (!f) {
    throw new Error(`unknown feature ${id}`);
  }
  return f;
}

export interface VoiceSettings {
  pitchMethod: 'rmvpe' | 'fcpe' | 'harvest' | 'dio' | 'pm';
  encoderPrecision: 'int8' | 'fp32';
  chunkSeconds: number;
  accelerator: 'auto' | 'cpu' | 'xnnpack' | 'nnapi' | 'qnn' | 'coreml';
  loadStrategy: 'resident' | 'sequential';
  pitchShift: number;
  rmsMixRate: number;
  crossfadeMs: number;
  lowMemory: boolean;
  threads: number;
}

/** Device-aware defaults: what we would pick for a user who changes nothing. */
export function recommendedVoiceSettings(p: DeviceProfile): VoiceSettings {
  const ramGB = p.totalRam / 1e9;
  return {
    pitchMethod: ramGB >= 6 ? 'rmvpe' : ramGB >= 4 ? 'fcpe' : 'dio',
    encoderPrecision: ramGB >= 10 ? 'fp32' : 'int8',
    chunkSeconds: p.cores >= 8 ? 2.5 : 3,
    accelerator: 'auto',
    loadStrategy: ramGB < 4 ? 'sequential' : 'resident',
    pitchShift: 0,
    rmsMixRate: 0.25,
    crossfadeMs: 60,
    lowMemory: ramGB < 6,
    threads: 0,
  };
}
