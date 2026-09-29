/** Snapshot of the hardware the app is running on. Pure data: easy to test. */
export interface DeviceProfile {
  platform: 'android' | 'ios' | 'other';
  osVersion: string;
  apiLevel: number; // Android API level, 0 on iOS
  model: string;
  brand: string;
  isEmulator: boolean;
  totalRam: number; // bytes
  availableRam: number; // bytes, at time of snapshot
  freeStorage: number; // bytes
  cores: number;
  maxFreqMhz: number;
  cpuFeatures: string[];
  hasFp16: boolean;
  hasDotProd: boolean;
  hasI8mm: boolean;
  soc: string;
  gpu: {name: string; type: string; adreno: boolean; mali: boolean; apple: boolean};
  hasNpu: boolean;
  abis: string[];
  /** Backend devices reported by llama.cpp (e.g. "CPU", "GPUOpenCL", "HTP0"). */
  llamaDevices: string[];
  voiceModuleAvailable: boolean;
  ortProviders: string[];
}

export type DeviceTier = 'entry' | 'low' | 'mid' | 'high' | 'flagship';

export const GB = 1e9;

export function deviceTier(p: Pick<DeviceProfile, 'totalRam' | 'cores'>): DeviceTier {
  const ram = p.totalRam / GB;
  if (ram < 3) {
    return 'entry';
  }
  if (ram < 5) {
    return 'low';
  }
  if (ram < 7.5) {
    return 'mid';
  }
  if (ram < 11.5 || p.cores < 8) {
    return 'high';
  }
  return 'flagship';
}

export const TIER_LABEL: Record<DeviceTier, string> = {
  entry: 'Entry (<3 GB)',
  low: 'Low (3–4 GB)',
  mid: 'Mid-range (6 GB)',
  high: 'High-end (8 GB+)',
  flagship: 'Flagship (12 GB+)',
};

/** Largest model (parameters, billions) we recommend at Q4 for a tier. */
export function recommendedMaxParamsB(tier: DeviceTier): number {
  return {entry: 0.6, low: 1.5, mid: 3.2, high: 4.5, flagship: 8.5}[tier];
}

/** Qualcomm SoC → Hexagon generation (HTP v73+ is supported by llama.cpp). */
const HEXAGON_SOCS: Array<[RegExp, string]> = [
  [/SM8850|8 Elite Gen 5/i, 'v81'],
  [/SM8750|8 Elite/i, 'v79'],
  [/SM8650|8 Gen 3/i, 'v75'],
  [/SM8550|8 Gen 2/i, 'v73'],
  [/SM8475|SM8450|8\+? Gen 1/i, 'v69'],
];

export function hexagonVersion(soc: string): string | null {
  for (const [re, v] of HEXAGON_SOCS) {
    if (re.test(soc)) {
      return v;
    }
  }
  return null;
}

/**
 * GPU offload support for llama.cpp:
 *  - Android: OpenCL backend is tuned for Adreno 7xx and needs dotprod + i8mm.
 *  - iOS: Metal needs an Apple7+ GPU (A14 / M1 or newer).
 */
export function gpuOffloadSupport(p: DeviceProfile): {supported: boolean; reason?: string} {
  if (p.isEmulator) {
    return {supported: false, reason: 'GPU offload is disabled on emulators/simulators'};
  }
  if (p.platform === 'ios') {
    const ok = p.gpu.apple && /Apple7|Apple8|Apple9|A1[4-9]|M\d/i.test(`${p.gpu.name} ${p.gpu.type}`);
    return ok ? {supported: true} : {supported: p.gpu.apple, reason: p.gpu.apple ? undefined : 'Metal GPU not detected'};
  }
  if (p.platform === 'android') {
    if (!p.gpu.adreno) {
      return {supported: false, reason: 'OpenCL offload currently targets Qualcomm Adreno GPUs'};
    }
    if (!p.hasDotProd || !p.hasI8mm) {
      return {supported: false, reason: 'CPU lacks dotprod/i8mm, required by the OpenCL build'};
    }
    return {supported: true};
  }
  return {supported: false, reason: 'Unsupported platform'};
}

export function npuSupport(p: DeviceProfile): {
  supported: boolean;
  experimental?: boolean;
  reason?: string;
  generation?: string;
} {
  if (p.platform === 'ios') {
    return {supported: false, reason: 'Apple Neural Engine is used by the voice engine (CoreML), not by llama.cpp'};
  }
  const gen = hexagonVersion(p.soc);
  const htpDevice = p.llamaDevices.some(d => d.startsWith('HTP'));
  if (!p.hasNpu && !htpDevice) {
    return {supported: false, reason: 'No Qualcomm Hexagon DSP detected'};
  }
  if (!gen && !htpDevice) {
    return {supported: false, reason: 'Hexagon NPU offload needs Snapdragon 8 Gen 1 (SM8450) or newer'};
  }
  if (gen === 'v69') {
    return {
      supported: true,
      experimental: true,
      generation: gen,
      reason: 'Snapdragon 8 Gen 1 (HTP v69) is at the edge of llama.cpp support — expect fallbacks to CPU',
    };
  }
  return {supported: true, experimental: true, generation: gen ?? 'HTP'};
}
