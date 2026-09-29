import * as FS from '@dr.pogodin/react-native-fs';
import AsyncStorage from '@react-native-async-storage/async-storage';
import {makeAutoObservable, runInAction} from 'mobx';
import {makePersistable} from 'mobx-persist-store';
import {PermissionsAndroid, Platform} from 'react-native';
import {
  ConversionStats,
  EngineInfo,
  getVoiceEngine,
  LiveStats,
  ModelInspection,
  TtsVoice,
  VoiceBenchmark,
  VoiceEngineConfig,
  VoiceJsi,
  VoiceNative,
} from 'react-native-mobigpt-voice';

import bundledVoices from '../../catalog/voices.json';
import {VoiceSettings} from '../features/registry';
import {DownloadManager} from '../services/downloads';
import {HFClient, matchFile} from '../services/hf';
import {pickAndImport} from '../services/importFile';
import {ensureDirs, Paths, safeFileName} from '../services/paths';
import {uid} from '../utils/format';
import type {SettingsStore} from './SettingsStore';

export interface VoiceAssetPreset {
  id: string;
  name: string;
  repo: string;
  file: string;
  sizeBytes: number;
  precision: 'int8' | 'fp32' | 'fp16';
  method?: 'rmvpe' | 'fcpe';
  channels?: number;
  license?: string;
  notes?: string;
  /** Direct download URLs tried before the Hugging Face file (e.g. MobiGPT's own INT8 builds). */
  urls?: string[];
  /** Parity vs FP32 measured in CI (tools/rvc/quantize_rvc.py sweep). */
  fidelity?: {cosMean: number; cosP5: number};
}

export interface VoicePreset {
  id: string;
  name: string;
  description?: string;
  repo: string;
  file: string;
  sampleRate?: number;
  version?: string;
  license?: string;
  /** Approximate, for display; the real size comes from the Hub listing. */
  sizeBytes?: number;
}

export interface InstalledAsset {
  id: string;
  kind: 'encoder' | 'pitch';
  name: string;
  path: string;
  sizeBytes: number;
  precision: string;
  method?: 'rmvpe' | 'fcpe';
}

export interface VoiceEntry {
  id: string;
  name: string;
  path: string;
  sizeBytes: number;
  sampleRate: number;
  version: string;
  usesF0: boolean;
  layout: string;
  source: 'preset' | 'import' | 'hf';
  repo?: string;
  description?: string;
  addedAt: number;
}

export interface ConversionRecord {
  id: string;
  voiceName: string;
  inputLabel: string;
  inputPath: string;
  outputPath: string;
  stats: ConversionStats;
  createdAt: number;
}

type VoicesCatalog = {
  version: number;
  encoders: VoiceAssetPreset[];
  pitch: VoiceAssetPreset[];
  voices: VoicePreset[];
  search: {query: string; hint: string};
};

export const voicesCatalog = bundledVoices as unknown as VoicesCatalog;

export class VoiceStore {
  assets: InstalledAsset[] = [];
  voices: VoiceEntry[] = [];
  selectedVoiceId: string | null = null;
  conversions: ConversionRecord[] = [];

  engineInfo: EngineInfo | null = null;
  engineKey: string | null = null;
  engineLoading = false;
  converting = false;
  progress = 0;
  recording = false;
  recordLevel = 0;
  lastRecording: string | null = null;
  playingPath: string | null = null;
  live: LiveStats | null = null;
  liveRunning = false;
  ttsVoices: TtsVoice[] = [];
  error: string | null = null;
  lastBenchmark: VoiceBenchmark | null = null;

  readonly downloads: DownloadManager;
  private hf: HFClient;
  private timers: ReturnType<typeof setInterval>[] = [];

  constructor(private settings: SettingsStore, private engineProvider: () => Promise<VoiceJsi | null> = getVoiceEngine, persist = true) {
    this.hf = new HFClient(settings.hfToken || undefined);
    this.downloads = new DownloadManager(() => this.hf.headers());
    makeAutoObservable(this, {downloads: false}, {autoBind: true});
    if (persist) {
      makePersistable(this, {
        name: 'mobigpt.voice.v1',
        properties: ['assets', 'voices', 'selectedVoiceId', 'conversions'],
        storage: AsyncStorage,
      }).catch(() => undefined);
    }
  }

  // ------------------------------------------------------------ selectors

  get voiceSettings(): VoiceSettings {
    if (!this.settings.voice) {
      throw new Error('Voice settings not initialised');
    }
    return this.settings.voice;
  }

  get selectedVoice(): VoiceEntry | undefined {
    return this.voices.find(v => v.id === this.selectedVoiceId) ?? this.voices[0];
  }

  isAssetInstalled(id: string): boolean {
    return this.assets.some(a => a.id === id);
  }

  get encoderAsset(): InstalledAsset | undefined {
    const enc = this.assets.filter(a => a.kind === 'encoder');
    const want = this.settings.voice?.encoderPrecision ?? 'int8';
    return enc.find(a => a.precision === want) ?? enc[0];
  }

  pitchAsset(method: 'rmvpe' | 'fcpe'): InstalledAsset | undefined {
    const list = this.assets.filter(a => a.kind === 'pitch' && a.method === method);
    const want = this.settings.voice?.encoderPrecision ?? 'int8';
    return list.find(a => a.precision === want) ?? list[0];
  }

  /** What is still missing for the current settings (empty = ready). */
  get missing(): string[] {
    const out: string[] = [];
    if (!this.encoderAsset) {
      out.push('Content encoder (HuBERT/ContentVec)');
    }
    const m = this.settings.voice?.pitchMethod;
    if ((m === 'rmvpe' || m === 'fcpe') && !this.pitchAsset(m)) {
      out.push(`${m.toUpperCase()} pitch model`);
    }
    if (!this.selectedVoice) {
      out.push('A voice model');
    }
    return out;
  }

  get ready(): boolean {
    return this.missing.length === 0;
  }

  get installedBytes(): number {
    return [...this.assets, ...this.voices].reduce((s, a) => s + (a.sizeBytes || 0), 0);
  }

  buildConfig(): VoiceEngineConfig {
    const v = this.voiceSettings;
    const voice = this.selectedVoice;
    const enc = this.encoderAsset;
    if (!voice || !enc) {
      throw new Error(`Missing: ${this.missing.join(', ')}`);
    }
    const method = v.pitchMethod;
    const pitchModel = method === 'rmvpe' || method === 'fcpe' ? this.pitchAsset(method)?.path : undefined;
    if ((method === 'rmvpe' || method === 'fcpe') && !pitchModel) {
      throw new Error(`Install the ${method.toUpperCase()} pitch model or pick a DSP pitch extractor.`);
    }
    return {
      encoderPath: enc.path,
      voicePath: voice.path,
      pitchMethod: method,
      pitchModelPath: pitchModel,
      accelerator: v.accelerator,
      threads: v.threads,
      lowMemory: v.lowMemory,
      loadStrategy: v.loadStrategy,
      pitchShift: v.pitchShift,
      rmsMixRate: v.rmsMixRate,
      indexRate: 0, // FAISS retrieval is never used on device
      chunkSeconds: v.chunkSeconds,
      crossfadeMs: v.crossfadeMs,
      defaultSampleRate: voice.sampleRate || 40000,
    };
  }

  // ------------------------------------------------------------ installing

  async installAsset(preset: VoiceAssetPreset, kind: 'encoder' | 'pitch') {
    await ensureDirs();
    this.hf.setToken(this.settings.hfToken);
    let fetched: {path: string; size: number} | null = null;
    let lastError: unknown;
    // Preferred mirrors first (MobiGPT-built INT8), then the Hugging Face source.
    for (const url of preset.urls ?? []) {
      try {
        const dest = `${Paths.voiceBase}/${safeFileName(`${preset.id}__${url.split('/').pop()}`)}`;
        await this.downloads.start(preset.id, url, dest, preset.name, 0);
        fetched = {path: dest, size: (await FS.stat(dest)).size};
        break;
      } catch (e) {
        lastError = e;
        this.downloads.clear(preset.id);
      }
    }
    if (!fetched && preset.repo) {
      fetched = await this.fetchFromHf(preset.id, preset.repo, preset.file, preset.name, preset.sizeBytes, Paths.voiceBase);
    }
    if (!fetched) {
      throw lastError instanceof Error ? lastError : new Error('No download source available');
    }
    const {path, size} = fetched;
    runInAction(() => {
      this.assets = this.assets.filter(a => a.id !== preset.id);
      this.assets.push({id: preset.id, kind, name: preset.name, path, sizeBytes: size, precision: preset.precision, method: preset.method});
    });
  }

  async uninstallAsset(id: string) {
    const a = this.assets.find(x => x.id === id);
    if (a) {
      await this.unloadEngine();
      await FS.unlink(a.path).catch(() => undefined);
      runInAction(() => (this.assets = this.assets.filter(x => x.id !== id)));
    }
  }

  async installPresetVoice(p: VoicePreset) {
    await ensureDirs();
    this.hf.setToken(this.settings.hfToken);
    const {path, size} = await this.fetchFromHf(p.id, p.repo, p.file, p.name, 0, Paths.voices);
    await this.registerVoice(path, size, {id: p.id, name: p.name, source: 'preset', repo: p.repo, description: p.description});
  }

  async addHfVoice(repo: string, file: string) {
    await ensureDirs();
    const id = `hf:${repo}/${file}`;
    const {path, size} = await this.fetchFromHf(id, repo, file, file, 0, Paths.voices);
    const name = file
      .split('/')
      .pop()!
      .replace(/\.onnx$/i, '');
    await this.registerVoice(path, size, {id, name, source: 'hf', repo});
  }

  async importVoice(): Promise<VoiceEntry | null> {
    await ensureDirs();
    const f = await pickAndImport(Paths.voices, ['.onnx']);
    if (!f) {
      return null;
    }
    return this.registerVoice(f.path, f.size, {id: uid('v_'), name: f.name.replace(/\.onnx$/i, ''), source: 'import'});
  }

  /** Validates a synthesiser with the native inspector before adding it. */
  async registerVoice(
    path: string,
    size: number,
    meta: {id: string; name: string; source: VoiceEntry['source']; repo?: string; description?: string},
  ): Promise<VoiceEntry> {
    const engine = await this.requireEngine();
    let info: ModelInspection;
    try {
      info = await engine.inspect(path);
    } catch (e: any) {
      await FS.unlink(path).catch(() => undefined);
      throw new Error(`Not a valid ONNX model: ${e?.message ?? e}`);
    }
    if (info.kind !== 'synthesizer' || !info.voice) {
      await FS.unlink(path).catch(() => undefined);
      throw new Error(
        info.kind === 'encoder' || info.kind === 'rmvpe' || info.kind === 'fcpe'
          ? `This file is a ${info.kind} model, not a voice. Install it from the "Engine packs" section instead.`
          : 'This ONNX file is not an RVC voice (expected inputs "phone" or "feats"). Export your .pth with RVC\'s ONNX tab or tools/rvc/export_voice_onnx.py.',
      );
    }
    const entry: VoiceEntry = {
      id: meta.id,
      name: meta.name,
      path,
      sizeBytes: size,
      sampleRate: info.voice.sampleRate || 40000,
      version: info.voice.version,
      usesF0: info.voice.usesF0,
      layout: info.voice.layout,
      source: meta.source,
      repo: meta.repo,
      description: meta.description,
      addedAt: Date.now(),
    };
    runInAction(() => {
      this.voices = this.voices.filter(v => v.id !== entry.id);
      this.voices.push(entry);
      this.selectedVoiceId = entry.id;
    });
    return entry;
  }

  async removeVoice(id: string) {
    const v = this.voices.find(x => x.id === id);
    if (!v) {
      return;
    }
    if (this.selectedVoiceId === id) {
      await this.unloadEngine();
    }
    await FS.unlink(v.path).catch(() => undefined);
    runInAction(() => {
      this.voices = this.voices.filter(x => x.id !== id);
      if (this.selectedVoiceId === id) {
        this.selectedVoiceId = this.voices[0]?.id ?? null;
      }
    });
  }

  selectVoice(id: string) {
    this.selectedVoiceId = id;
  }

  private async fetchFromHf(id: string, repo: string, file: string, label: string, expected: number, dir: string) {
    let resolved = file;
    let size = expected;
    try {
      const f = matchFile(await this.hf.listFiles(repo), file);
      if (!f) {
        throw new Error(`${file} was not found in ${repo}. The upstream repository may have changed.`);
      }
      resolved = f.path;
      size = f.size;
    } catch (e: any) {
      if (/not found/.test(e?.message)) {
        throw e;
      }
    }
    const dest = `${dir}/${safeFileName(`${repo.replace('/', '__')}__${resolved.split('/').pop()}`)}`;
    await this.downloads.start(id, this.hf.resolveUrl(repo, resolved), dest, label, size);
    const stat = await FS.stat(dest);
    return {path: dest, size: stat.size};
  }

  async searchHf(query: string) {
    this.hf.setToken(this.settings.hfToken);
    return this.hf.search(query || voicesCatalog.search.query, 'onnx', 30);
  }

  async listOnnx(repo: string) {
    const files = await this.hf.listFiles(repo);
    return files.filter(f => f.path.toLowerCase().endsWith('.onnx'));
  }

  // ---------------------------------------------------------------- engine

  private async requireEngine(): Promise<VoiceJsi> {
    const e = await this.engineProvider();
    if (!e) {
      throw new Error('The voice engine is not included in this build.');
    }
    return e;
  }

  async ensureEngine(): Promise<EngineInfo> {
    const cfg = this.buildConfig();
    const key = JSON.stringify({...cfg, pitchShift: 0});
    const engine = await this.requireEngine();
    if (this.engineKey === key && this.engineInfo && engine.isLoaded()) {
      engine.setPitchShift(cfg.pitchShift ?? 0);
      return this.engineInfo;
    }
    this.engineLoading = true;
    this.error = null;
    try {
      const info = await engine.load(cfg);
      runInAction(() => {
        this.engineInfo = info;
        this.engineKey = key;
      });
      return info;
    } catch (e: any) {
      runInAction(() => {
        this.error = e?.message ?? String(e);
        this.engineInfo = null;
        this.engineKey = null;
      });
      throw e;
    } finally {
      runInAction(() => (this.engineLoading = false));
    }
  }

  async unloadEngine() {
    await this.stopLive();
    const e = await this.engineProvider();
    await e?.unload();
    runInAction(() => {
      this.engineInfo = null;
      this.engineKey = null;
    });
  }

  // ----------------------------------------------------------- conversion

  /** Converts any audio file (decoded natively if it is not WAV). */
  async convert(inputPath: string, inputLabel: string): Promise<ConversionRecord> {
    await ensureDirs();
    this.converting = true;
    this.progress = 0;
    this.error = null;
    try {
      const engine = await this.requireEngine();
      await this.ensureEngine();
      let wav = inputPath;
      if (!/\.wav$/i.test(inputPath)) {
        if (!VoiceNative) {
          throw new Error('Audio decoder unavailable');
        }
        wav = `${Paths.temp}/decoded-${Date.now()}.wav`;
        await VoiceNative.decodeAudioToWav(inputPath, wav);
      }
      const voice = this.selectedVoice!;
      const out = `${Paths.outputs}/${safeFileName(voice.name)}-${Date.now()}.wav`;
      const stats = await engine.convertFile(wav, out, p => runInAction(() => (this.progress = p)));
      if (stats.cancelled) {
        await FS.unlink(out).catch(() => undefined);
        throw new Error('Conversion cancelled');
      }
      const rec: ConversionRecord = {
        id: uid('x_'),
        voiceName: voice.name,
        inputLabel,
        inputPath: wav,
        outputPath: out,
        stats,
        createdAt: Date.now(),
      };
      runInAction(() => {
        this.conversions.unshift(rec);
        this.conversions = this.conversions.slice(0, 50);
      });
      return rec;
    } catch (e: any) {
      runInAction(() => (this.error = e?.message ?? String(e)));
      throw e;
    } finally {
      runInAction(() => (this.converting = false));
    }
  }

  async cancelConversion() {
    (await this.engineProvider())?.cancel();
  }

  /** Renders text with the operating system's (offline) TTS into a WAV file. */
  async synthesizeSystem(text: string, ttsVoiceId = '', language = ''): Promise<string> {
    if (!VoiceNative) {
      throw new Error('Text-to-speech is unavailable in this build');
    }
    await ensureDirs();
    const wav = `${Paths.recordings}/tts-${Date.now()}.wav`;
    await VoiceNative.synthesizeSpeech(text, wav, language, ttsVoiceId, 1.0, 1.0);
    return wav;
  }

  async convertText(text: string, ttsVoiceId = '', language = ''): Promise<ConversionRecord> {
    const wav = await this.synthesizeSystem(text, ttsVoiceId, language);
    return this.convert(wav, `“${text.slice(0, 40)}${text.length > 40 ? '…' : ''}”`);
  }

  async loadTtsVoices() {
    if (!VoiceNative) {
      return;
    }
    const list = await VoiceNative.listTtsVoices();
    runInAction(() => (this.ttsVoices = list.filter(v => !v.requiresNetwork)));
  }

  async deleteConversion(id: string) {
    const r = this.conversions.find(c => c.id === id);
    if (r) {
      await FS.unlink(r.outputPath).catch(() => undefined);
    }
    runInAction(() => (this.conversions = this.conversions.filter(c => c.id !== id)));
  }

  // ----------------------------------------------------- mic / playback

  async ensureMicPermission(): Promise<boolean> {
    if (Platform.OS !== 'android') {
      return true; // iOS prompts natively on first capture
    }
    const res = await PermissionsAndroid.request(PermissionsAndroid.PERMISSIONS.RECORD_AUDIO, {
      title: 'Microphone',
      message: 'MobiGPT needs the microphone to record your voice. Audio never leaves the device.',
      buttonPositive: 'Allow',
    });
    return res === PermissionsAndroid.RESULTS.GRANTED;
  }

  async startRecording() {
    if (!(await this.ensureMicPermission())) {
      throw new Error('Microphone permission denied');
    }
    await ensureDirs();
    const engine = await this.requireEngine();
    const path = `${Paths.recordings}/rec-${Date.now()}.wav`;
    await engine.startRecording(path);
    runInAction(() => {
      this.recording = true;
      this.lastRecording = null;
    });
    this.timers.push(setInterval(() => runInAction(() => (this.recordLevel = engine.recordingLevel())), 100));
  }

  async stopRecording(): Promise<string | null> {
    const engine = await this.requireEngine();
    this.clearTimers();
    try {
      const r = await engine.stopRecording();
      runInAction(() => (this.lastRecording = r.path));
      return r.path;
    } finally {
      runInAction(() => {
        this.recording = false;
        this.recordLevel = 0;
      });
    }
  }

  async play(path: string) {
    const engine = await this.requireEngine();
    if (this.playingPath) {
      engine.stopPlayback();
    }
    runInAction(() => (this.playingPath = path));
    try {
      await engine.play(path);
    } finally {
      runInAction(() => {
        if (this.playingPath === path) {
          this.playingPath = null;
        }
      });
    }
  }

  async stopPlayback() {
    (await this.engineProvider())?.stopPlayback();
    runInAction(() => (this.playingPath = null));
  }

  async startLive() {
    if (!(await this.ensureMicPermission())) {
      throw new Error('Microphone permission denied');
    }
    if (this.voiceSettings.loadStrategy !== 'resident') {
      throw new Error('Live mode needs the "Resident" model loading option (Voice settings).');
    }
    const engine = await this.requireEngine();
    await this.ensureEngine();
    await engine.startLive();
    runInAction(() => (this.liveRunning = true));
    this.timers.push(
      setInterval(() => {
        const s = engine.liveStats();
        runInAction(() => {
          this.live = s;
          if (s.error) {
            this.error = s.error;
          }
        });
      }, 250),
    );
  }

  async stopLive() {
    if (!this.liveRunning) {
      return;
    }
    this.clearTimers();
    const engine = await this.engineProvider();
    await engine?.stopLive();
    runInAction(() => (this.liveRunning = false));
  }

  setPitchShift(semitones: number) {
    this.settings.updateVoice({pitchShift: semitones});
    this.engineProvider().then(e => e?.setPitchShift(semitones));
  }

  async benchmark(seconds = 5): Promise<VoiceBenchmark> {
    const engine = await this.requireEngine();
    await this.ensureEngine();
    const r = await engine.benchmark(seconds);
    runInAction(() => (this.lastBenchmark = r));
    return r;
  }

  private clearTimers() {
    this.timers.forEach(clearInterval);
    this.timers = [];
  }
}
