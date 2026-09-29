import AsyncStorage from '@react-native-async-storage/async-storage';
import {makeAutoObservable} from 'mobx';
import {makePersistable} from 'mobx-persist-store';

import {FeatureId, VoiceSettings} from '../features/registry';
import {Accel} from '../services/llm';

export type ThemeMode = 'system' | 'light' | 'dark';

export interface LlmSettings {
  nCtx: number;
  nThreads: number; // 0 = auto
  accel: Accel;
  gpuLayers: number;
  flashAttn: boolean;
  cacheType: 'f16' | 'q8_0' | 'q4_0';
  useMlock: boolean;
  temperature: number;
  topP: number;
  topK: number;
  minP: number;
  maxTokens: number;
  repeatPenalty: number;
  systemPrompt: string;
  enableThinking: boolean;
}

export const DEFAULT_LLM: LlmSettings = {
  nCtx: 2048,
  nThreads: 0,
  accel: 'cpu',
  gpuLayers: 99,
  flashAttn: false,
  cacheType: 'f16',
  useMlock: false,
  temperature: 0.7,
  topP: 0.95,
  topK: 40,
  minP: 0.05,
  maxTokens: 1024,
  repeatPenalty: 1.1,
  systemPrompt: 'You are MobiGPT, a helpful assistant running entirely on this device. Answer concisely.',
  enableThinking: false,
};

export class SettingsStore {
  themeMode: ThemeMode = 'system';
  onboardingDone = false;
  hfToken = '';
  llm: LlmSettings = {...DEFAULT_LLM};
  voice: VoiceSettings | null = null; // null until device-aware defaults are applied
  enabledFeatures: FeatureId[] = ['chat', 'benchmarks'];
  lastModelId: string | null = null;
  autoLoadLastModel = true;
  checkUpdatesOnLaunch = true;
  hydrated = false;

  constructor(persist = true) {
    makeAutoObservable(this, {}, {autoBind: true});
    if (persist) {
      makePersistable(this, {
        name: 'mobigpt.settings.v1',
        properties: [
          'themeMode',
          'onboardingDone',
          'hfToken',
          'llm',
          'voice',
          'enabledFeatures',
          'lastModelId',
          'autoLoadLastModel',
          'checkUpdatesOnLaunch',
        ],
        storage: AsyncStorage,
      })
        .then(() => this.setHydrated())
        .catch(() => this.setHydrated());
    } else {
      this.hydrated = true;
    }
  }

  setHydrated() {
    this.hydrated = true;
    this.llm = {...DEFAULT_LLM, ...this.llm};
  }

  setTheme(m: ThemeMode) {
    this.themeMode = m;
  }

  completeOnboarding() {
    this.onboardingDone = true;
  }

  resetOnboarding() {
    this.onboardingDone = false;
  }

  setHfToken(t: string) {
    this.hfToken = t.trim();
  }

  updateLlm(patch: Partial<LlmSettings>) {
    this.llm = {...this.llm, ...patch};
  }

  resetLlm() {
    this.llm = {...DEFAULT_LLM};
  }

  updateVoice(patch: Partial<VoiceSettings>) {
    if (this.voice) {
      this.voice = {...this.voice, ...patch};
    }
  }

  setVoice(v: VoiceSettings) {
    this.voice = v;
  }

  isEnabled(id: FeatureId): boolean {
    return this.enabledFeatures.includes(id);
  }

  setFeature(id: FeatureId, on: boolean) {
    const s = new Set(this.enabledFeatures);
    if (on) {
      s.add(id);
    } else {
      s.delete(id);
    }
    this.enabledFeatures = Array.from(s);
  }

  setLastModel(id: string | null) {
    this.lastModelId = id;
  }

  setAutoLoad(v: boolean) {
    this.autoLoadLastModel = v;
  }

  setCheckUpdates(v: boolean) {
    this.checkUpdatesOnLaunch = v;
  }
}
