import AsyncStorage from '@react-native-async-storage/async-storage';
import {makeAutoObservable, runInAction} from 'mobx';
import {makePersistable} from 'mobx-persist-store';

import {getLlmEngine} from '../services/llm';
import {uid} from '../utils/format';
import type {DeviceStore} from './DeviceStore';
import type {ModelStore} from './ModelStore';
import type {SettingsStore} from './SettingsStore';
import type {VoiceStore} from './VoiceStore';

export interface BenchmarkResult {
  id: string;
  kind: 'llm' | 'voice';
  createdAt: number;
  device: string;
  subject: string; // model or voice name
  accel: string;
  // llm
  ppTps?: number;
  tgTps?: number;
  nCtx?: number;
  threads?: number;
  loadMs?: number;
  // voice
  rtf?: number;
  encoderMs?: number;
  pitchMs?: number;
  synthMs?: number;
  audioSeconds?: number;
  pitchMethod?: string;
}

export type BenchPreset = 'quick' | 'standard' | 'long';

export const BENCH_PRESETS: Record<BenchPreset, {pp: number; tg: number; nr: number; label: string}> = {
  quick: {pp: 128, tg: 32, nr: 1, label: 'Quick (pp128 / tg32)'},
  standard: {pp: 512, tg: 128, nr: 3, label: 'Standard (pp512 / tg128 ×3)'},
  long: {pp: 1024, tg: 256, nr: 3, label: 'Long (pp1024 / tg256 ×3)'},
};

export class BenchmarkStore {
  results: BenchmarkResult[] = [];
  running: 'llm' | 'voice' | null = null;
  error: string | null = null;

  constructor(
    private settings: SettingsStore,
    private device: DeviceStore,
    private models: ModelStore,
    private voice: VoiceStore,
    persist = true,
  ) {
    makeAutoObservable(this, {}, {autoBind: true});
    if (persist) {
      makePersistable(this, {name: 'mobigpt.bench.v1', properties: ['results'], storage: AsyncStorage}).catch(() => undefined);
    }
  }

  get best(): {pp: number; tg: number; rtf: number} {
    const llm = this.results.filter(r => r.kind === 'llm');
    const v = this.results.filter(r => r.kind === 'voice' && r.rtf);
    return {
      pp: Math.max(0, ...llm.map(r => r.ppTps ?? 0)),
      tg: Math.max(0, ...llm.map(r => r.tgTps ?? 0)),
      rtf: v.length ? Math.min(...v.map(r => r.rtf!)) : 0,
    };
  }

  async runLlm(preset: BenchPreset = 'standard'): Promise<BenchmarkResult> {
    const model = this.models.loaded;
    if (!model) {
      throw new Error('Load a model on the Models tab first.');
    }
    this.running = 'llm';
    this.error = null;
    try {
      const p = BENCH_PRESETS[preset];
      const r = await getLlmEngine().bench(p.pp, p.tg, 1, p.nr);
      const result: BenchmarkResult = {
        id: uid('b_'),
        kind: 'llm',
        createdAt: Date.now(),
        device: this.device.profile.model,
        subject: model.name,
        accel: this.models.loadedInfo?.gpu
          ? `${this.settings.llm.accel.toUpperCase()} (${(this.models.loadedInfo.devices ?? []).join(', ') || 'offload'})`
          : 'CPU',
        ppTps: r.ppTps,
        tgTps: r.tgTps,
        threads: r.nThreads,
        nCtx: this.settings.llm.nCtx,
        loadMs: this.models.lastLoadMs,
      };
      runInAction(() => this.results.unshift(result));
      return result;
    } catch (e: any) {
      runInAction(() => (this.error = e?.message ?? String(e)));
      throw e;
    } finally {
      runInAction(() => (this.running = null));
    }
  }

  async runVoice(seconds = 6): Promise<BenchmarkResult> {
    this.running = 'voice';
    this.error = null;
    try {
      const r = await this.voice.benchmark(seconds);
      const result: BenchmarkResult = {
        id: uid('b_'),
        kind: 'voice',
        createdAt: Date.now(),
        device: this.device.profile.model,
        subject: this.voice.selectedVoice?.name ?? 'voice',
        accel: r.providers.join(', '),
        rtf: r.realtimeFactor,
        encoderMs: r.stages.encoderMs,
        pitchMs: r.stages.pitchMs,
        synthMs: r.stages.synthMs,
        audioSeconds: r.audioSeconds,
        pitchMethod: this.settings.voice?.pitchMethod,
      };
      runInAction(() => this.results.unshift(result));
      return result;
    } catch (e: any) {
      runInAction(() => (this.error = e?.message ?? String(e)));
      throw e;
    } finally {
      runInAction(() => (this.running = null));
    }
  }

  clear() {
    this.results = [];
  }

  remove(id: string) {
    this.results = this.results.filter(r => r.id !== id);
  }
}
