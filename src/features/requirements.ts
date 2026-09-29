import {DeviceProfile, GB} from './device';

export type Severity = 'ok' | 'info' | 'warn' | 'block';

export interface Requirement {
  platforms?: Array<'android' | 'ios'>;
  minAndroidApi?: number;
  minIosMajor?: number;
  /** Below this the feature is blocked. */
  minRamGB?: number;
  /** Below this the user is warned (feature still allowed). */
  recommendedRamGB?: number;
  minCores?: number;
  recommendedCores?: number;
  /** Big-core clock that keeps real-time audio under 1x RTF. */
  recommendedMaxFreqMhz?: number;
  /** Bytes of free storage needed to install. */
  storageBytes?: number;
  cpuFeaturesRecommended?: string[];
  nativeModule?: 'voice';
  warnOnEmulator?: boolean;
  /** Arbitrary extra check. */
  custom?: (p: DeviceProfile) => Issue | null;
}

export interface Issue {
  severity: Exclude<Severity, 'ok'>;
  code: string;
  message: string;
}

export interface Evaluation {
  status: Severity;
  issues: Issue[];
}

const RANK: Record<Severity, number> = {ok: 0, info: 1, warn: 2, block: 3};

export function worst(a: Severity, b: Severity): Severity {
  return RANK[a] >= RANK[b] ? a : b;
}

export function evaluate(req: Requirement, p: DeviceProfile): Evaluation {
  const issues: Issue[] = [];
  const push = (severity: Issue['severity'], code: string, message: string) => issues.push({severity, code, message});
  const ramGB = p.totalRam / GB;

  if (req.platforms && p.platform !== 'other' && !req.platforms.includes(p.platform)) {
    push('block', 'platform', `Not available on ${p.platform === 'ios' ? 'iOS' : 'Android'}.`);
  }
  if (req.minAndroidApi && p.platform === 'android' && p.apiLevel > 0 && p.apiLevel < req.minAndroidApi) {
    push('block', 'os', `Requires Android API ${req.minAndroidApi}+ (this device: ${p.apiLevel}).`);
  }
  if (req.minIosMajor && p.platform === 'ios' && parseInt(p.osVersion, 10) < req.minIosMajor) {
    push('block', 'os', `Requires iOS ${req.minIosMajor}+ (this device: ${p.osVersion}).`);
  }
  if (req.nativeModule === 'voice' && !p.voiceModuleAvailable) {
    push('block', 'native', 'This build was compiled without the voice engine (mobigpt.features.json → voice.enabled).');
  }
  if (req.minRamGB && ramGB < req.minRamGB) {
    push('block', 'ram', `Needs at least ${req.minRamGB} GB RAM (this device: ${ramGB.toFixed(1)} GB).`);
  } else if (req.recommendedRamGB && ramGB < req.recommendedRamGB) {
    push(
      'warn',
      'ram',
      `${req.recommendedRamGB} GB RAM recommended (this device: ${ramGB.toFixed(
        1,
      )} GB). Expect slowdowns or the OS closing the app under memory pressure.`,
    );
  }
  if (req.minCores && p.cores < req.minCores) {
    push('block', 'cpu', `Needs ${req.minCores}+ CPU cores (this device: ${p.cores}).`);
  } else if (req.recommendedCores && p.cores < req.recommendedCores) {
    push('warn', 'cpu', `${req.recommendedCores}+ CPU cores recommended (this device: ${p.cores}).`);
  }
  if (req.recommendedMaxFreqMhz && p.maxFreqMhz > 0 && p.maxFreqMhz < req.recommendedMaxFreqMhz) {
    push(
      'warn',
      'freq',
      `A ${(req.recommendedMaxFreqMhz / 1000).toFixed(1)} GHz+ big core is recommended (this device peaks at ${(
        p.maxFreqMhz / 1000
      ).toFixed(1)} GHz).`,
    );
  }
  if (req.storageBytes && p.freeStorage > 0 && p.freeStorage < req.storageBytes * 1.1) {
    push(
      'block',
      'storage',
      `Needs ${(req.storageBytes / 1e6).toFixed(0)} MB free storage (available: ${(p.freeStorage / 1e6).toFixed(0)} MB).`,
    );
  }
  if (req.cpuFeaturesRecommended?.length) {
    const missing = req.cpuFeaturesRecommended.filter(f => !hasFeature(p, f));
    if (missing.length) {
      push('info', 'cpu-features', `CPU lacks ${missing.join(', ')}; inference will use slower kernels.`);
    }
  }
  if (req.warnOnEmulator && p.isEmulator) {
    push('info', 'emulator', 'Running on an emulator: performance numbers are not representative.');
  }
  const extra = req.custom?.(p);
  if (extra) {
    issues.push(extra);
  }
  const status = issues.reduce<Severity>((s, i) => worst(s, i.severity), 'ok');
  return {status, issues};
}

export function hasFeature(p: DeviceProfile, f: string): boolean {
  switch (f) {
    case 'dotprod':
      return p.hasDotProd;
    case 'i8mm':
      return p.hasI8mm;
    case 'fp16':
      return p.hasFp16;
    default:
      return p.cpuFeatures.includes(f);
  }
}

/**
 * Memory budget check for a concrete load (model + runtime), against RAM the
 * OS will realistically give us. Android keeps ~40 % of RAM for itself and
 * background apps; iOS caps a single app at roughly 50-65 %.
 */
export function memoryFit(requiredBytes: number, p: DeviceProfile): {severity: Severity; message: string} {
  const usable = Math.max(p.availableRam, p.totalRam * (p.platform === 'ios' ? 0.55 : 0.6));
  const ratio = requiredBytes / usable;
  if (ratio <= 0.75) {
    return {severity: 'ok', message: 'Fits comfortably in memory.'};
  }
  if (ratio <= 1.0) {
    return {severity: 'warn', message: 'Tight fit: close other apps before loading.'};
  }
  if (requiredBytes < p.totalRam * 0.85) {
    return {severity: 'warn', message: 'Likely exceeds free memory; the OS may kill the app. Try a smaller quantisation or context.'};
  }
  return {severity: 'block', message: "Exceeds this device's RAM."};
}
