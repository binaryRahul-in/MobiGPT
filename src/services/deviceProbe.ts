import * as FS from '@dr.pogodin/react-native-fs';
import {Platform} from 'react-native';
import DeviceInfo from 'react-native-device-info';
import {HardwareInfo} from 'react-native-mobigpt-device';
import {getVoiceEngine, isVoiceModuleAvailable} from 'react-native-mobigpt-voice';

import {DeviceProfile} from '../features/device';
import {getLlmEngine} from './llm';

async function safe<T>(p: Promise<T> | undefined, fallback: T): Promise<T> {
  try {
    return p ? await p : fallback;
  } catch {
    return fallback;
  }
}

/** Collects everything the requirement engine needs. Never throws. */
export async function probeDevice(): Promise<DeviceProfile> {
  const [totalRam, model, brand, isEmulator, apiLevel, fsInfo, cpu, gpu, avail, npu, llamaDevices] = await Promise.all([
    safe(DeviceInfo.getTotalMemory(), 0),
    safe(Promise.resolve(DeviceInfo.getModel()), 'Unknown'),
    safe(Promise.resolve(DeviceInfo.getBrand()), ''),
    safe(DeviceInfo.isEmulator(), false),
    Platform.OS === 'android' ? safe(DeviceInfo.getApiLevel(), 0) : Promise.resolve(0),
    safe(FS.getFSInfo(), {freeSpace: 0, totalSpace: 0, freeSpaceEx: 0, totalSpaceEx: 0}),
    safe(HardwareInfo?.getCPUInfo(), undefined),
    safe(HardwareInfo?.getGPUInfo(), undefined),
    safe(HardwareInfo?.getAvailableMemory(), 0),
    safe(HardwareInfo?.hasNpu(), false),
    safe(getLlmEngine().backendDevices(), [] as string[]),
  ]);
  let ortProviders: string[] = [];
  const voiceAvailable = isVoiceModuleAvailable();
  if (voiceAvailable) {
    const v = await safe(getVoiceEngine(), null);
    ortProviders = v ? safe2(() => v.version().providers, []) : [];
  }
  return {
    platform: Platform.OS === 'ios' ? 'ios' : Platform.OS === 'android' ? 'android' : 'other',
    osVersion: String(Platform.Version),
    apiLevel,
    model,
    brand,
    isEmulator,
    totalRam,
    availableRam: avail || totalRam * 0.4,
    freeStorage: fsInfo.freeSpace,
    cores: cpu?.cores ?? 4,
    maxFreqMhz: cpu?.maxFreqMhz ?? 0,
    cpuFeatures: cpu?.features ?? [],
    hasFp16: cpu?.hasFp16 ?? false,
    hasDotProd: cpu?.hasDotProd ?? false,
    hasI8mm: cpu?.hasI8mm ?? false,
    soc: [cpu?.socModel, cpu?.hardware].filter(Boolean).join(' / '),
    gpu: {
      name: gpu?.renderer ?? '',
      type: gpu?.gpuType ?? 'Unknown',
      adreno: gpu?.hasAdreno ?? false,
      mali: gpu?.hasMali ?? false,
      apple: gpu?.hasAppleGpu ?? false,
    },
    hasNpu: npu,
    abis: cpu?.abis ?? [],
    llamaDevices,
    voiceModuleAvailable: voiceAvailable,
    ortProviders,
  };
}

function safe2<T>(fn: () => T, fallback: T): T {
  try {
    return fn();
  } catch {
    return fallback;
  }
}
