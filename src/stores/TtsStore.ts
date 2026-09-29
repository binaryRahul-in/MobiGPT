/**
 * On-device neural text-to-speech: Kokoro-82M (ONNX Runtime, C++) driven by the
 * TypeScript G2P in services/tts. Only phoneme-token ids cross into native code;
 * audio is written straight to a WAV file.
 */
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as FS from '@dr.pogodin/react-native-fs';
import {makeAutoObservable, runInAction} from 'mobx';
import {makePersistable} from 'mobx-persist-store';
import {getVoiceEngine, TtsResult, VoiceJsi} from 'react-native-mobigpt-voice';

import {DownloadManager} from '../services/downloads';
import {HFClient} from '../services/hf';
import {ensureDirs, Paths, safeFileName} from '../services/paths';
import {EnglishG2P, Lexicon, toWindows} from '../services/tts/g2p';
import type {SettingsStore} from './SettingsStore';
import {voicesCatalog} from './VoiceStore';

export interface TtsModelPreset {
  id: string;
  name: string;
  precision: string;
  sizeBytes: number;
  license: string;
  sampleRate: number;
  urls: string[];
  repo: string;
  file: string;
  notes?: string;
}

export interface TtsVoicePreset {
  id: string;
  name: string;
  accent: string;
  gender: 'female' | 'male';
  description?: string;
  sizeBytes: number;
  urls: string[];
  repo: string;
  file: string;
}

export interface TtsCatalog {
  models: TtsModelPreset[];
  voices: TtsVoicePreset[];
  lexicon: {id: string; name: string; license: string; sizeBytes: number; files: {name: string; urls: string[]}[]};
}

export const ttsCatalog = (voicesCatalog as unknown as {tts: TtsCatalog}).tts;

export interface InstallStep {
  index: number;
  total: number;
  id: string;
  label: string;
}

interface PlanItem {
  id: string;
  label: string;
  bytes: number;
  run: () => Promise<void>;
}

export class TtsStore {
  modelPath: string | null = null;
  modelId: string | null = null;
  lexiconPaths: string[] = [];
  voicePaths: Record<string, string> = {};
  selectedVoiceId = 'af_heart';
  speed = 1.0;
  engine: 'neural' | 'system' = 'neural';
  convertWithRvc = true;

  installing: InstallStep | null = null;
  installError: string | null = null;
  busy = false;
  error: string | null = null;
  lastResult: TtsResult | null = null;
  lastUnknown: string[] = [];

  readonly downloads: DownloadManager;
  private hf: HFClient;
  private g2p: EnglishG2P | null = null;

  constructor(private settings: SettingsStore, private engineProvider: () => Promise<VoiceJsi | null> = getVoiceEngine, persist = true) {
    this.hf = new HFClient(settings.hfToken || undefined);
    this.downloads = new DownloadManager(() => this.hf.headers());
    makeAutoObservable(this, {downloads: false}, {autoBind: true});
    if (persist) {
      makePersistable(this, {
        name: 'mobigpt.tts.v1',
        properties: ['modelPath', 'modelId', 'lexiconPaths', 'voicePaths', 'selectedVoiceId', 'speed', 'engine', 'convertWithRvc'],
        storage: AsyncStorage,
      }).catch(() => undefined);
    }
  }

  get model(): TtsModelPreset {
    return ttsCatalog.models[0];
  }

  get selectedVoice(): TtsVoicePreset {
    return ttsCatalog.voices.find(v => v.id === this.selectedVoiceId) ?? ttsCatalog.voices[0];
  }

  get ready(): boolean {
    return !!this.modelPath && this.lexiconPaths.length === ttsCatalog.lexicon.files.length && !!this.voicePaths[this.selectedVoiceId];
  }

  /** What still needs downloading for the selected voice, in install order. */
  get plan(): PlanItem[] {
    const items: PlanItem[] = [];
    if (!this.modelPath) {
      items.push({id: this.model.id, label: `Speech model · ${this.model.name}`, bytes: this.model.sizeBytes, run: this.installModel});
    }
    if (this.lexiconPaths.length !== ttsCatalog.lexicon.files.length) {
      items.push({
        id: ttsCatalog.lexicon.id,
        label: 'Pronunciation dictionary',
        bytes: ttsCatalog.lexicon.sizeBytes,
        run: this.installLexicon,
      });
    }
    const v = this.selectedVoice;
    if (!this.voicePaths[v.id]) {
      items.push({id: `tts-voice:${v.id}`, label: `Voice · ${v.name}`, bytes: v.sizeBytes, run: () => this.installVoice(v.id)});
    }
    return items;
  }

  get missingBytes(): number {
    return this.plan.reduce((a, p) => a + p.bytes, 0);
  }

  get installedBytes(): number {
    return (
      (this.modelPath ? this.model.sizeBytes : 0) +
      (this.lexiconPaths.length ? ttsCatalog.lexicon.sizeBytes : 0) +
      Object.keys(this.voicePaths).length * 522240
    );
  }

  setVoice(id: string) {
    this.selectedVoiceId = id;
  }

  setSpeed(speed: number) {
    this.speed = Math.min(2, Math.max(0.5, speed));
  }

  setEngine(engine: 'neural' | 'system') {
    this.engine = engine;
  }

  setConvertWithRvc(on: boolean) {
    this.convertWithRvc = on;
  }

  // ------------------------------------------------------------ install

  async install() {
    this.installError = null;
    const steps = this.plan;
    try {
      for (let i = 0; i < steps.length; i++) {
        runInAction(() => (this.installing = {index: i + 1, total: steps.length, id: steps[i].id, label: steps[i].label}));
        await steps[i].run();
      }
    } catch (e: any) {
      runInAction(() => (this.installError = e?.message ?? String(e)));
      throw e;
    } finally {
      runInAction(() => (this.installing = null));
    }
  }

  async installModel() {
    const m = this.model;
    const path = await this.fetch(m.id, m.name, m.urls, m.repo, m.file, 'model.onnx');
    await this.unload();
    runInAction(() => {
      this.modelPath = path;
      this.modelId = m.id;
    });
  }

  async installLexicon() {
    const lex = ttsCatalog.lexicon;
    const paths: string[] = [];
    for (const f of lex.files) {
      paths.push(await this.fetch(lex.id, `${lex.name} (${f.name})`, f.urls, null, null, f.name));
    }
    this.g2p = null;
    runInAction(() => (this.lexiconPaths = paths));
  }

  async installVoice(id: string) {
    const v = ttsCatalog.voices.find(x => x.id === id);
    if (!v) {
      throw new Error(`Unknown voice ${id}`);
    }
    const path = await this.fetch(`tts-voice:${v.id}`, `Voice ${v.name}`, v.urls, v.repo, v.file, `voice_${v.id}.bin`);
    runInAction(() => (this.voicePaths = {...this.voicePaths, [v.id]: path}));
  }

  async uninstall() {
    await this.unload();
    const files = [this.modelPath, ...this.lexiconPaths, ...Object.values(this.voicePaths)].filter((p): p is string => !!p);
    await Promise.all(files.map(p => FS.unlink(p).catch(() => undefined)));
    runInAction(() => {
      this.modelPath = null;
      this.modelId = null;
      this.lexiconPaths = [];
      this.voicePaths = {};
      this.lastResult = null;
    });
  }

  /** Mirrors first, then the Hugging Face copy. */
  private async fetch(id: string, label: string, urls: string[], repo: string | null, file: string | null, name: string): Promise<string> {
    await ensureDirs();
    const dest = `${Paths.tts}/${safeFileName(name)}`;
    const sources = [...urls, ...(repo && file ? [this.hf.resolveUrl(repo, file)] : [])];
    let last: unknown = new Error('No download source');
    for (const url of sources) {
      try {
        await this.downloads.start(id, url, dest, label, 0);
        return dest;
      } catch (e) {
        last = e;
        if (/cancelled/i.test((e as Error)?.message ?? '')) {
          throw e;
        }
        this.downloads.clear(id);
      }
    }
    throw last;
  }

  // ------------------------------------------------------------ synthesis

  private async loadG2P(): Promise<EnglishG2P> {
    if (!this.g2p) {
      const [gold, silver] = await Promise.all(this.lexiconPaths.map(p => FS.readFile(p, 'utf8').then(t => JSON.parse(t) as Lexicon)));
      this.g2p = new EnglishG2P(gold, silver ?? {});
    }
    return this.g2p;
  }

  private async engineReady(): Promise<VoiceJsi> {
    const engine = await this.engineProvider();
    if (!engine) {
      throw new Error('The voice engine is not included in this build.');
    }
    if (!engine.ttsIsLoaded()) {
      if (!this.modelPath) {
        throw new Error('Install the neural voice first.');
      }
      const info = await engine.ttsLoad(this.modelPath, {accelerator: 'auto', lowMemory: this.settings.voice?.lowMemory ?? false});
      info.warnings.forEach(w => console.warn(`[tts] ${w}`));
    }
    return engine;
  }

  /** Text -> 24 kHz WAV on disk. */
  async synthesize(text: string): Promise<{path: string; result: TtsResult}> {
    if (!text.trim()) {
      throw new Error('Type something to say first.');
    }
    if (!this.ready) {
      throw new Error('Install the neural voice first.');
    }
    this.busy = true;
    this.error = null;
    try {
      const engine = await this.engineReady();
      const g2p = await this.loadG2P();
      const {phonemes, unknown} = g2p.phonemize(text);
      const windows = toWindows(phonemes);
      if (windows.length === 0) {
        throw new Error('Nothing to say: the text has no readable words.');
      }
      await ensureDirs();
      const out = `${Paths.recordings}/tts-${Date.now()}.wav`;
      const result = await engine.ttsSynthesize({
        windows: windows.map(w => w.tokens),
        pauses: windows.map(w => w.pauseAfter),
        voicePath: this.voicePaths[this.selectedVoiceId],
        speed: this.speed,
        outputPath: out,
      });
      runInAction(() => {
        this.lastResult = result;
        this.lastUnknown = unknown;
      });
      return {path: out, result};
    } catch (e: any) {
      runInAction(() => (this.error = e?.message ?? String(e)));
      throw e;
    } finally {
      runInAction(() => (this.busy = false));
    }
  }

  /** Frees the model and the lexicon (~100 MB) until the next synthesis. */
  async unload() {
    this.g2p = null;
    const engine = await this.engineProvider();
    if (engine?.ttsIsLoaded()) {
      await engine.ttsUnload();
    }
  }
}
