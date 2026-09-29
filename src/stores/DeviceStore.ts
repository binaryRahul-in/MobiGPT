import {makeAutoObservable, runInAction} from 'mobx';

import {DeviceProfile, DeviceTier, deviceTier, gpuOffloadSupport, npuSupport} from '../features/device';
import {FEATURES, FeatureDefinition, FeatureId} from '../features/registry';
import {evaluate, Evaluation} from '../features/requirements';

export const FALLBACK_PROFILE: DeviceProfile = {
  platform: 'android',
  osVersion: '0',
  apiLevel: 0,
  model: 'Unknown device',
  brand: '',
  isEmulator: false,
  totalRam: 4e9,
  availableRam: 1.6e9,
  freeStorage: 0,
  cores: 4,
  maxFreqMhz: 0,
  cpuFeatures: [],
  hasFp16: false,
  hasDotProd: false,
  hasI8mm: false,
  soc: '',
  gpu: {name: '', type: 'Unknown', adreno: false, mali: false, apple: false},
  hasNpu: false,
  abis: [],
  llamaDevices: [],
  voiceModuleAvailable: false,
  ortProviders: [],
};

export class DeviceStore {
  profile: DeviceProfile = FALLBACK_PROFILE;
  probed = false;
  probing = false;
  lastProbe = 0;

  constructor(private probe: () => Promise<DeviceProfile>) {
    makeAutoObservable(this, {}, {autoBind: true});
  }

  async refresh(): Promise<DeviceProfile> {
    if (this.probing) {
      return this.profile;
    }
    this.probing = true;
    try {
      const p = await this.probe();
      runInAction(() => {
        this.profile = p;
        this.probed = true;
        this.lastProbe = Date.now();
      });
      return p;
    } finally {
      runInAction(() => {
        this.probing = false;
      });
    }
  }

  get tier(): DeviceTier {
    return deviceTier(this.profile);
  }

  get gpu() {
    return gpuOffloadSupport(this.profile);
  }

  get npu() {
    return npuSupport(this.profile);
  }

  evaluation(f: FeatureDefinition | FeatureId): Evaluation {
    const def = typeof f === 'string' ? FEATURES.find(x => x.id === f)! : f;
    return evaluate(def.requirement, this.profile);
  }
}
