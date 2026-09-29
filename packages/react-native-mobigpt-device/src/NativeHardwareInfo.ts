import type {TurboModule} from 'react-native';
import {TurboModuleRegistry} from 'react-native';

// Adapted from PocketPal AI (MIT, (c) Asghar Ghorbani) — see NOTICE.md.
export interface CPUInfo {
  cores: number;
  features: string[];
  hasFp16: boolean;
  hasDotProd: boolean;
  hasSve: boolean;
  hasI8mm: boolean;
  socModel: string;
  hardware: string;
  maxFreqMhz: number;
  abis: string[];
}

export interface GPUInfo {
  renderer: string;
  vendor: string;
  version: string;
  hasAdreno: boolean;
  hasMali: boolean;
  hasPowerVR: boolean;
  hasAppleGpu: boolean;
  gpuType: string;
}

export interface Spec extends TurboModule {
  getCPUInfo(): Promise<CPUInfo>;
  getGPUInfo(): Promise<GPUInfo>;
  getAvailableMemory(): Promise<number>;
  /** Android: "libcdsprpc.so" presence (Hexagon DSP / NPU). iOS: Neural Engine presence. */
  hasNpu(): Promise<boolean>;
}

export default TurboModuleRegistry.get<Spec>('HardwareInfo');
