import * as FS from '@dr.pogodin/react-native-fs';
import AsyncStorage from '@react-native-async-storage/async-storage';
import {makeAutoObservable, runInAction} from 'mobx';
import {makePersistable} from 'mobx-persist-store';

import bundledCatalog from '../../catalog/models.json';
import {DeviceProfile} from '../features/device';
import {memoryFit, Severity} from '../features/requirements';
import {DownloadManager} from '../services/downloads';
import {HFClient, HFFile, matchFile} from '../services/hf';
import {pickAndImport} from '../services/importFile';
import {Paths, safeFileName} from '../services/paths';
import {getLlmEngine, LoadedModelInfo} from '../services/llm';
import {fetchCatalog} from '../services/updates';
import {detectQuant, estimateModelMemory, GGUFMetadata, parseLlamaModelInfo, Quant} from '../utils/gguf';
import type {SettingsStore} from './SettingsStore';

export interface CatalogModel {
  id: string;
  name: string;
  family?: string;
  paramsB?: number;
  repo: string;
  file: string;
  sizeBytes: number;
  contextLength?: number;
  license?: string;
  tags?: string[];
  description?: string;
}

export interface ModelEntry extends Omit<CatalogModel, 'repo'> {
  repo?: string;
  source: 'catalog' | 'hf' | 'import';
  localPath?: string;
  meta?: GGUFMetadata;
  addedAt: number;
}

export interface Compatibility {
  severity: Severity;
  message: string;
  requiredBytes: number;
}

type ModelsCatalog = {version: number; models: CatalogModel[]};

export class ModelStore {
  /** Everything that has been downloaded, imported or added from HF. */
  local: ModelEntry[] = [];
  catalog: CatalogModel[] = (bundledCatalog as ModelsCatalog).models;
  catalogSource: 'bundled' | 'remote' = 'bundled';

  loadedId: string | null = null;
  loadedInfo: LoadedModelInfo | null = null;
  loading = false;
  loadProgress = 0;
  loadError: string | null = null;
  lastLoadMs = 0;

  readonly downloads: DownloadManager;
  readonly hf: HFClient;

  constructor(private settings: SettingsStore, private device: () => DeviceProfile, persist = true) {
    this.hf = new HFClient(settings.hfToken || undefined);
    this.downloads = new DownloadManager(() => this.hf.headers());
    makeAutoObservable(this, {downloads: false, hf: false}, {autoBind: true});
    if (persist) {
      makePersistable(this, {name: 'mobigpt.models.v1', properties: ['local'], storage: AsyncStorage}).catch(() => undefined);
    }
  }

  // ------------------------------------------------------------- listings

  /** Catalog entries merged with local state, plus HF/imported extras. */
  get all(): ModelEntry[] {
    const byId = new Map(this.local.map(m => [m.id, m]));
    const merged: ModelEntry[] = this.catalog.map(c => byId.get(c.id) ?? {...c, source: 'catalog', addedAt: 0});
    for (const m of this.local) {
      if (!this.catalog.some(c => c.id === m.id)) {
        merged.push(m);
      }
    }
    return merged;
  }

  get downloaded(): ModelEntry[] {
    return this.local.filter(m => !!m.localPath);
  }

  get loaded(): ModelEntry | undefined {
    return this.local.find(m => m.id === this.loadedId);
  }

  byId(id: string): ModelEntry | undefined {
    return this.all.find(m => m.id === id);
  }

  quant(m: ModelEntry): Quant {
    return detectQuant(m.file);
  }

  compatibility(m: ModelEntry): Compatibility {
    const p = this.device();
    const required = estimateModelMemory({
      fileSizeBytes: m.sizeBytes,
      nCtx: this.settings.llm.nCtx,
      cacheTypeK: this.settings.llm.cacheType,
      cacheTypeV: this.settings.llm.flashAttn ? this.settings.llm.cacheType : 'f16',
      meta: m.meta,
    });
    const fit = memoryFit(required, p);
    return {severity: fit.severity, message: fit.message, requiredBytes: required};
  }

  /** Catalog models sorted "best for this device first". */
  get recommended(): ModelEntry[] {
    return [...this.all]
      .filter(m => m.source === 'catalog' && this.compatibility(m).severity === 'ok')
      .sort((a, b) => (b.paramsB ?? 0) - (a.paramsB ?? 0))
      .slice(0, 3);
  }

  // ------------------------------------------------------------- catalog

  async refreshCatalog() {
    const {catalog, source} = await fetchCatalog<ModelsCatalog>('models', bundledCatalog as ModelsCatalog);
    runInAction(() => {
      this.catalog = catalog.models;
      this.catalogSource = source;
    });
  }

  // ------------------------------------------------------------ downloads

  private upsert(entry: ModelEntry) {
    const i = this.local.findIndex(m => m.id === entry.id);
    if (i >= 0) {
      this.local[i] = entry;
    } else {
      this.local.push(entry);
    }
  }

  async download(id: string): Promise<void> {
    const m = this.byId(id);
    if (!m?.repo) {
      throw new Error('Unknown model');
    }
    this.hf.setToken(this.settings.hfToken);
    // Resolve against the live repo listing: catches renamed files and gives the true size.
    let file: HFFile | undefined;
    try {
      file = matchFile(await this.hf.listFiles(m.repo), m.file);
    } catch {
      file = undefined; // offline listing failure: try the direct URL anyway
    }
    const fileName = file?.path ?? m.file;
    const size = file?.size ?? m.sizeBytes;
    const p = this.device();
    if (p.freeStorage > 0 && size * 1.05 > p.freeStorage) {
      throw new Error(`Not enough storage: needs ${(size / 1e9).toFixed(2)} GB free.`);
    }
    const dest = `${Paths.models}/${safeFileName(`${m.repo.replace('/', '__')}__${fileName.split('/').pop()}`)}`;
    await this.downloads.start(id, this.hf.resolveUrl(m.repo, fileName), dest, m.name, size);
    runInAction(() => this.upsert({...m, file: fileName, sizeBytes: size, localPath: dest, addedAt: Date.now()}));
    await this.readMetadata(id);
  }

  cancelDownload(id: string) {
    this.downloads.cancel(id);
  }

  /** Adds an arbitrary GGUF from a Hugging Face repo (from the HF browser). */
  addFromHf(repo: string, file: HFFile, displayName?: string): string {
    const id = `hf:${repo}/${file.path}`;
    if (!this.local.some(m => m.id === id)) {
      this.local.push({
        id,
        name:
          displayName ??
          `${repo
            .split('/')
            .pop()
            ?.replace(/-GGUF$/i, '')} · ${detectQuant(file.path)}`,
        repo,
        file: file.path,
        sizeBytes: file.size,
        source: 'hf',
        addedAt: Date.now(),
        tags: ['hugging face'],
      });
    }
    return id;
  }

  async importFromDevice(): Promise<string | null> {
    const f = await pickAndImport(Paths.models, ['.gguf']);
    if (!f) {
      return null;
    }
    const id = `import:${f.name}:${Date.now()}`;
    runInAction(() =>
      this.local.push({
        id,
        name: f.name.replace(/\.gguf$/i, ''),
        file: f.name,
        sizeBytes: f.size,
        source: 'import',
        localPath: f.path,
        addedAt: Date.now(),
        tags: ['imported'],
      }),
    );
    await this.readMetadata(id);
    return id;
  }

  async readMetadata(id: string) {
    const m = this.local.find(x => x.id === id);
    if (!m?.localPath) {
      return;
    }
    try {
      const info = await getLlmEngine().readModelInfo(m.localPath);
      const meta = parseLlamaModelInfo(info);
      runInAction(() => {
        const cur = this.local.find(x => x.id === id);
        if (cur) {
          cur.meta = meta;
          if (meta?.contextLength) {
            cur.contextLength = meta.contextLength;
          }
        }
      });
    } catch {
      // Metadata is an optimisation for the memory estimate only.
    }
  }

  async remove(id: string) {
    if (this.loadedId === id) {
      await this.unload();
    }
    const m = this.local.find(x => x.id === id);
    if (m?.localPath && (await FS.exists(m.localPath))) {
      await FS.unlink(m.localPath);
    }
    runInAction(() => {
      // HF entries stay listed (so they can be re-downloaded); catalog and
      // imported entries disappear from the local list entirely.
      this.local = this.local
        .filter(x => x.id !== id || x.source === 'hf')
        .map(x => (x.id === id ? {...x, localPath: undefined, meta: undefined} : x));
      if (this.settings.lastModelId === id) {
        this.settings.setLastModel(null);
      }
    });
  }

  // ----------------------------------------------------------- load/unload

  async load(id: string, opts: {force?: boolean} = {}): Promise<void> {
    const m = this.local.find(x => x.id === id);
    if (!m?.localPath) {
      throw new Error('Download the model first');
    }
    if (!(await FS.exists(m.localPath))) {
      runInAction(() => (m.localPath = undefined));
      throw new Error('Model file is missing; download it again');
    }
    const compat = this.compatibility(m);
    if (compat.severity === 'block' && !opts.force) {
      throw new Error(`Not enough memory: ~${(compat.requiredBytes / 1e9).toFixed(1)} GB needed. ${compat.message}`);
    }
    this.loading = true;
    this.loadProgress = 0;
    this.loadError = null;
    const s = this.settings.llm;
    const t0 = Date.now();
    try {
      const info = await getLlmEngine().load(
        {
          path: m.localPath,
          nCtx: s.nCtx,
          nThreads: s.nThreads,
          accel: s.accel,
          gpuLayers: s.gpuLayers,
          flashAttn: s.flashAttn,
          cacheType: s.cacheType,
          useMlock: s.useMlock,
        },
        p => runInAction(() => (this.loadProgress = p)),
      );
      runInAction(() => {
        this.loadedId = id;
        this.loadedInfo = info;
        this.lastLoadMs = Date.now() - t0;
        this.settings.setLastModel(id);
      });
    } catch (e: any) {
      runInAction(() => {
        this.loadError = e?.message ?? String(e);
        this.loadedId = null;
        this.loadedInfo = null;
      });
      throw e;
    } finally {
      runInAction(() => {
        this.loading = false;
      });
    }
  }

  async unload() {
    await getLlmEngine().unload();
    runInAction(() => {
      this.loadedId = null;
      this.loadedInfo = null;
    });
  }

  async autoLoad() {
    const id = this.settings.lastModelId;
    if (this.settings.autoLoadLastModel && id && !this.loadedId && this.local.some(m => m.id === id && m.localPath)) {
      await this.load(id).catch(() => undefined);
    }
  }
}
