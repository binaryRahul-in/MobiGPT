import * as fs from '../jest/mocks/fs';
import {fakeEngine, inspections, VoiceNative} from '../jest/mocks/voice';
import {recommendedVoiceSettings} from '../src/features/registry';
import {BenchOutcome, ChatMessage, GenerationParams, LlmEngine, LoadOptions, setLlmEngine} from '../src/services/llm';
import {RootStore} from '../src/stores/RootStore';
import {voicesCatalog} from '../src/stores/VoiceStore';
import {profile} from '../jest/fixtures';

class FakeLlm implements LlmEngine {
  loaded: LoadOptions | null = null;
  stopped = false;
  async backendDevices() {
    return ['CPU', 'GPUOpenCL'];
  }
  async readModelInfo() {
    return {
      'general.architecture': 'qwen3',
      'qwen3.embedding_length': 1024,
      'qwen3.attention.head_count': 16,
      'qwen3.attention.head_count_kv': 8,
      'qwen3.block_count': 28,
      'qwen3.context_length': 40960,
      'qwen3.vocab_size': 151936,
    };
  }
  async load(o: LoadOptions, onProgress?: (p: number) => void) {
    onProgress?.(0.5);
    onProgress?.(1);
    this.loaded = o;
    return {
      description: 'qwen3 0.6B',
      sizeBytes: 4e8,
      nParams: 6e8,
      gpu: o.accel === 'gpu',
      reasonNoGPU: '',
      devices: [],
      systemInfo: '',
      metadata: {},
    };
  }
  async unload() {
    this.loaded = null;
  }
  isLoaded() {
    return !!this.loaded;
  }
  async chat(messages: ChatMessage[], p: GenerationParams, onToken: (t: {token: string}) => void) {
    this.lastMessages = messages;
    this.lastParams = p;
    for (const tok of ['Hello', ' from', ' MobiGPT']) {
      onToken({token: tok});
    }
    return {
      text: 'Hello from MobiGPT',
      reasoning: '',
      tokensPredicted: 3,
      tokensEvaluated: 12,
      promptTps: 120,
      genTps: 24,
      interrupted: false,
      contextFull: false,
    };
  }
  lastMessages: ChatMessage[] = [];
  lastParams?: GenerationParams;
  async stop() {
    this.stopped = true;
  }
  async bench(): Promise<BenchOutcome> {
    return {ppTps: 300, tgTps: 25, nThreads: 6, nGpuLayers: 0, flashAttn: false};
  }
}

let llm: FakeLlm;

const TREE: Array<[string, number]> = [
  ['Qwen3-0.6B-Q4_K_M.gguf', 397000000],
  ['contentvec_768l12_q8.onnx', 95200000],
  ['contentvec_768l12.onnx', 377800000],
  ['rmvpe_q8.onnx', 98700000],
  ['woman_1.onnx', 55000000],
];

function makeStore(over = {}) {
  const store = new RootStore({persist: false, probe: async () => profile(over)});
  // HF listings are served by the fake fetch below.
  return store;
}

beforeEach(() => {
  fs.__reset();
  llm = new FakeLlm();
  setLlmEngine(llm);
  for (const [name, size] of TREE) {
    fs.__setRemoteName(name, size);
  }
  (global as any).fetch = jest.fn(async (url: string) => {
    if (url.includes('/tree/')) {
      return {
        ok: true,
        status: 200,
        json: async () => TREE.map(([path, size]) => ({type: 'file', path, lfs: {size}})),
      };
    }
    return {ok: false, status: 404, json: async () => ({})};
  });
  Object.values(fakeEngine).forEach(v => (typeof v === 'function' && 'mockClear' in v ? (v as jest.Mock).mockClear() : null));
});

describe('ModelStore', () => {
  it('downloads a catalog model through HF, reads metadata and loads it', async () => {
    const s = makeStore();
    await s.bootstrap();
    fs.__setRemote('https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf', 397000000);
    await s.models.download('qwen3-0.6b');
    const m = s.models.local.find(x => x.id === 'qwen3-0.6b')!;
    expect(m.localPath).toMatch(/models\/unsloth__Qwen3-0.6B-GGUF__Qwen3-0.6B-Q4_K_M.gguf$/);
    expect(m.meta?.nLayers).toBe(28);
    expect(s.models.downloads.get('qwen3-0.6b')?.state).toBe('done');

    await s.models.load('qwen3-0.6b');
    expect(s.models.loadedId).toBe('qwen3-0.6b');
    expect(llm.loaded?.nCtx).toBe(2048);
    expect(s.settings.lastModelId).toBe('qwen3-0.6b');

    await s.models.unload();
    expect(s.models.loadedId).toBeNull();
    await s.models.remove('qwen3-0.6b');
    expect(s.models.downloaded).toHaveLength(0);
  });

  it('refuses models that cannot fit unless forced', async () => {
    const s = makeStore({totalRam: 3e9, availableRam: 1e9});
    await s.bootstrap();
    fs.__putFile('/docs/big.gguf', 2.5e9);
    s.models.local.push({
      id: 'big',
      name: 'Big',
      file: 'big-Q4_K_M.gguf',
      sizeBytes: 2.5e9,
      source: 'import',
      localPath: '/docs/big.gguf',
      addedAt: 0,
    });
    expect(s.models.compatibility(s.models.local[0]).severity).toBe('block');
    await expect(s.models.load('big')).rejects.toThrow(/Not enough memory/);
    await s.models.load('big', {force: true});
    expect(s.models.loadedId).toBe('big');
  });

  it('surfaces HTTP errors from downloads', async () => {
    const s = makeStore();
    await s.bootstrap();
    fs.__setRemote('https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf', 10, 401);
    await expect(s.models.download('qwen3-0.6b')).rejects.toThrow(/gated/);
    expect(s.models.downloads.get('qwen3-0.6b')?.state).toBe('error');
  });

  it('adds models from HF and ranks recommendations by fit', async () => {
    const s = makeStore();
    await s.bootstrap();
    const id = s.models.addFromHf('org/Cool-GGUF', {path: 'cool-Q4_K_M.gguf', size: 5e8, quant: 'Q4_K_M'});
    expect(s.models.byId(id)?.name).toContain('Cool');
    expect(s.models.recommended.length).toBeGreaterThan(0);
    expect(s.models.recommended.every(m => s.models.compatibility(m).severity === 'ok')).toBe(true);
  });
});

describe('ChatStore', () => {
  it('streams a reply and records stats', async () => {
    const s = makeStore();
    await s.bootstrap();
    fs.__putFile('/docs/m.gguf', 4e8);
    s.models.local.push({
      id: 'm',
      name: 'Tiny',
      file: 'm-Q4_K_M.gguf',
      sizeBytes: 4e8,
      source: 'import',
      localPath: '/docs/m.gguf',
      addedAt: 0,
    });
    await s.models.load('m');
    await s.chat.send('Hi there');
    const conv = s.chat.active!;
    expect(conv.title).toBe('Hi there');
    expect(conv.messages.map(m => m.role)).toEqual(['user', 'assistant']);
    expect(conv.messages[1].content).toBe('Hello from MobiGPT');
    expect(conv.messages[1].stats?.genTps).toBe(24);
    expect(llm.lastMessages[0].role).toBe('system');
    expect(llm.lastMessages[llm.lastMessages.length - 1]).toEqual({role: 'user', content: 'Hi there'});
  });

  it('requires a loaded model and trims history to the context', async () => {
    const s = makeStore();
    await s.bootstrap();
    await expect(s.chat.send('hello')).rejects.toThrow(/Load a model/);
    const conv = s.chat.newConversation();
    for (let i = 0; i < 200; i++) {
      conv.messages.push({id: `u${i}`, role: i % 2 ? 'assistant' : 'user', content: 'x'.repeat(200), createdAt: i});
    }
    s.settings.updateLlm({nCtx: 1024, maxTokens: 128});
    const msgs = s.chat.buildMessages(conv);
    const chars = msgs.reduce((n, m) => n + m.content.length, 0);
    expect(chars).toBeLessThan(1024 * 3.2);
    expect(msgs.length).toBeGreaterThan(2);
  });
});

describe('BenchmarkStore', () => {
  it('runs llm and voice benchmarks and tracks the best results', async () => {
    const s = makeStore();
    await s.bootstrap();
    await expect(s.bench.runLlm()).rejects.toThrow(/Load a model/);
    fs.__putFile('/docs/m.gguf', 4e8);
    s.models.local.push({id: 'm', name: 'Tiny', file: 'm.gguf', sizeBytes: 4e8, source: 'import', localPath: '/docs/m.gguf', addedAt: 0});
    await s.models.load('m');
    await s.bench.runLlm('quick');
    expect(s.bench.best.tg).toBe(25);
    expect(s.bench.results[0].kind).toBe('llm');
  });
});

describe('VoiceStore', () => {
  async function readyVoiceStore() {
    const s = makeStore();
    await s.bootstrap();
    s.settings.setFeature('voice', true);
    s.settings.setVoice({...recommendedVoiceSettings(profile()), pitchMethod: 'rmvpe', encoderPrecision: 'int8'});
    return s;
  }

  it('reports missing packs, installs them from HF and builds an engine config', async () => {
    const s = await readyVoiceStore();
    expect(s.voice.missing).toEqual(['Content encoder (HuBERT/ContentVec)', 'RMVPE pitch model', 'A voice model']);
    const enc = voicesCatalog.encoders.find(e => e.id === 'contentvec-int8')!;
    const rmvpe = voicesCatalog.pitch.find(p => p.id === 'rmvpe-int8')!;
    await s.voice.installAsset(enc, 'encoder');
    await s.voice.installAsset(rmvpe, 'pitch');
    await s.voice.installPresetVoice(voicesCatalog.voices[0]);
    expect(s.voice.ready).toBe(true);
    const cfg = s.voice.buildConfig();
    expect(cfg.indexRate).toBe(0);
    expect(cfg.pitchMethod).toBe('rmvpe');
    expect(cfg.encoderPath).toContain('contentvec_768l12_q8.onnx');
    expect(cfg.pitchModelPath).toContain('rmvpe_q8.onnx');
    expect(cfg.chunkSeconds).toBeGreaterThanOrEqual(2);

    // DSP pitch needs no model file.
    s.settings.updateVoice({pitchMethod: 'dio'});
    expect(s.voice.buildConfig().pitchModelPath).toBeUndefined();
  });

  it('prefers the MobiGPT INT8 mirror and falls back to Hugging Face when it is unreachable', async () => {
    const s = await readyVoiceStore();
    const mobi = voicesCatalog.encoders.find(e => e.id === 'contentvec-int8-mobigpt')!;
    expect(mobi.urls?.length).toBeGreaterThan(0);
    await s.voice.installAsset(mobi, 'encoder');
    expect(s.voice.encoderAsset?.path).toContain('contentvec_768l12_int8_pc.onnx');

    await s.voice.uninstallAsset(mobi.id);
    fs.__setRemote(mobi.urls![0], 10, 404);
    await s.voice.installAsset(mobi, 'encoder');
    expect(s.voice.encoderAsset?.path).toContain('contentvec_768l12_q8.onnx');
  });

  it('rejects ONNX files that are not RVC voices', async () => {
    const s = await readyVoiceStore();
    inspections.set('voice.onnx', {
      kind: 'encoder',
      sizeMB: 1,
      quantized: false,
      inputs: [],
      outputs: [],
      metadata: {},
    });
    await expect(s.voice.importVoice()).rejects.toThrow(/encoder model, not a voice/);
    expect(s.voice.voices).toHaveLength(0);
    inspections.clear();
    const v = await s.voice.importVoice();
    expect(v?.sampleRate).toBe(40000);
    expect(s.voice.selectedVoice?.id).toBe(v?.id);
  });

  it('converts recordings, files and text; reloads the engine only when config changes', async () => {
    const s = await readyVoiceStore();
    s.settings.updateVoice({pitchMethod: 'pm'});
    await s.voice.installAsset(voicesCatalog.encoders[0], 'encoder');
    await s.voice.installPresetVoice(voicesCatalog.voices[0]);

    await s.voice.startRecording();
    expect(s.voice.recording).toBe(true);
    const rec = await s.voice.stopRecording();
    const r1 = await s.voice.convert(rec!, 'Recording');
    expect(r1.stats.realtimeFactor).toBeLessThan(1);
    expect(fakeEngine.load).toHaveBeenCalledTimes(1);

    const r2 = await s.voice.convert('/docs/song.mp3', 'song.mp3');
    expect(VoiceNative.decodeAudioToWav).toHaveBeenCalled();
    expect(r2.inputPath).toMatch(/decoded-.*\.wav$/);
    expect(fakeEngine.load).toHaveBeenCalledTimes(1); // same config → engine reused

    s.voice.setPitchShift(5);
    await s.voice.convertText('hello world');
    expect(fakeEngine.load).toHaveBeenCalledTimes(1); // pitch shift is applied live
    expect(fakeEngine.setPitchShift).toHaveBeenCalledWith(5);
    expect(s.voice.conversions).toHaveLength(3);

    s.settings.updateVoice({chunkSeconds: 3});
    await s.voice.convert(rec!, 'again');
    expect(fakeEngine.load).toHaveBeenCalledTimes(2);

    await s.voice.deleteConversion(s.voice.conversions[0].id);
    expect(s.voice.conversions).toHaveLength(3);
  });

  it('runs live mode only with the resident strategy', async () => {
    const s = await readyVoiceStore();
    s.settings.updateVoice({pitchMethod: 'dio', loadStrategy: 'sequential'});
    await s.voice.installAsset(voicesCatalog.encoders[0], 'encoder');
    await s.voice.installPresetVoice(voicesCatalog.voices[0]);
    await expect(s.voice.startLive()).rejects.toThrow(/Resident/);
    s.settings.updateVoice({loadStrategy: 'resident'});
    await s.voice.startLive();
    expect(s.voice.liveRunning).toBe(true);
    await s.voice.stopLive();
    expect(s.voice.liveRunning).toBe(false);
  });

  it('benchmarks voice RTF', async () => {
    const s = await readyVoiceStore();
    s.settings.updateVoice({pitchMethod: 'pm'});
    await s.voice.installAsset(voicesCatalog.encoders[0], 'encoder');
    await s.voice.installPresetVoice(voicesCatalog.voices[0]);
    const r = await s.bench.runVoice(4);
    expect(r.rtf).toBe(0.25);
    expect(s.bench.best.rtf).toBe(0.25);
  });
});

describe('DownloadManager', () => {
  const {DownloadManager, STALL_TIMEOUT_MS} = require('../src/services/downloads');

  beforeEach(() => fs.__reset());
  afterEach(() => jest.useRealTimers());

  it('aborts a download that stops receiving data instead of hanging', async () => {
    jest.useFakeTimers();
    const url = 'https://example.com/stalled.onnx';
    fs.__setStalled(url);
    const dm = new DownloadManager();
    const p = dm.start('x', url, '/docs/m/stalled.onnx', 'Speech encoder', 0);
    const settled = p.then(
      () => 'resolved',
      (e: Error) => e.message,
    );
    await jest.advanceTimersByTimeAsync(STALL_TIMEOUT_MS + 10_000);
    expect(await settled).toMatch(/Speech encoder: no data received for 45 s/);
    expect(dm.get('x')?.state).toBe('error');
    expect(dm.isActive('x')).toBe(false);
    expect(fs.__files().has('/docs/m/stalled.onnx.part')).toBe(false);
  });

  it('keeps a slow but progressing download alive', async () => {
    const dm = new DownloadManager();
    fs.__setRemote('https://example.com/ok.onnx', 5000);
    await expect(dm.start('y', 'https://example.com/ok.onnx', '/docs/m/ok.onnx', 'ok', 5000)).resolves.toBe('/docs/m/ok.onnx');
    expect(dm.get('y')?.state).toBe('done');
  });
});

describe('TtsStore', () => {
  const {TtsStore, ttsCatalog} = require('../src/stores/TtsStore');
  const {ttsRequests} = require('../jest/mocks/voice');
  const {toTokens} = require('../src/services/tts/g2p');
  const lexicon = JSON.stringify(require('../jest/fixtures-lexicon.json'));

  const make = () => {
    const root = new RootStore({persist: false, probe: async () => profile()});
    return new TtsStore(root.settings, undefined, false);
  };

  beforeEach(() => {
    fs.__reset();
    ttsRequests.length = 0;
    fs.__setRemoteText('misaki_us_gold.json', lexicon);
    fs.__setRemoteText('misaki_us_silver.json', '{}');
  });

  it('plans the model, dictionary and selected voice, then installs them in order', async () => {
    const tts = make();
    expect(tts.ready).toBe(false);
    expect(tts.plan.map((p: {label: string}) => p.label)).toEqual([
      'Speech model · Kokoro-82M v1.0',
      'Pronunciation dictionary',
      'Voice · Heart',
    ]);
    expect(tts.missingBytes).toBeGreaterThan(90e6);
    await tts.install();
    expect(tts.ready).toBe(true);
    expect(tts.installing).toBeNull();
    expect(tts.plan).toEqual([]);
    expect(tts.modelPath).toBe('/docs/mobigpt/tts/model.onnx');
  });

  it('falls back to the next source when the mirror is missing', async () => {
    const [mirror, upstream] = ttsCatalog.models[0].urls;
    fs.__setRemote(mirror, 0, 404);
    fs.__setRemote(upstream, 92361271);
    const tts = make();
    await tts.installModel();
    expect(tts.modelPath).toBeTruthy();
    expect(tts.downloads.get(ttsCatalog.models[0].id)?.url).toBe(upstream);
  });

  it('keeps the error on screen when every source fails', async () => {
    const tts = make();
    for (const u of ttsCatalog.models[0].urls) {
      fs.__setRemote(u, 0, 404);
    }
    fs.__setRemote('https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model_quantized.onnx', 0, 404);
    await expect(tts.install()).rejects.toThrow(/404/);
    expect(tts.installError).toMatch(/404/);
    expect(tts.ready).toBe(false);
  });

  it('turns text into phoneme-token windows for the native engine', async () => {
    const tts = make();
    await tts.install();
    tts.setSpeed(1.2);
    const {path, result} = await tts.synthesize('Hello world, this voice was made on my phone.');
    expect(path).toMatch(/\/voice\/recordings\/tts-\d+\.wav$/);
    expect(result.sampleRate).toBe(24000);
    expect(ttsRequests).toHaveLength(1);
    const req = ttsRequests[0];
    expect(req.windows).toEqual([toTokens('həlˈO wˈɜɹld, ðɪs vˈYs wʌz mˌAd ˌɔn mI fˈOn.')]);
    expect(req.speed).toBe(1.2);
    expect(req.voicePath).toBe('/docs/mobigpt/tts/voice_af_heart.bin');
    expect(tts.lastUnknown).toEqual([]);
    expect(tts.busy).toBe(false);
  });

  it('refuses to synthesise before installation or with empty text', async () => {
    const tts = make();
    await expect(tts.synthesize('Hello')).rejects.toThrow(/Install the neural voice/);
    await tts.install();
    await expect(tts.synthesize('   ')).rejects.toThrow(/Type something/);
  });

  it('switching voice only downloads that voice', async () => {
    const tts = make();
    await tts.install();
    tts.setVoice('am_michael');
    expect(tts.ready).toBe(false);
    expect(tts.plan.map((p: {id: string}) => p.id)).toEqual(['tts-voice:am_michael']);
  });
});

describe('VoiceStore · importing RVC voices', () => {
  const {rvcInfos, defaultRvcInfo} = require('../jest/mocks/voice');
  const TPL = 'https://github.com/binaryRahul-in/MobiGPT/releases/download/models-v1/rvc_template_v2_48k.onnx';
  const URL = 'https://huggingface.co/AIMan2001/PeterGriffin/resolve/main/Peter%20Griffin.zip';

  const make = () => new RootStore({persist: false, probe: async () => profile()}).voice;

  beforeEach(() => {
    fs.__reset();
    rvcInfos.clear();
    (fakeEngine.rvcImport as jest.Mock).mockClear();
    fs.__setRemote(URL, 75_952_056);
    fs.__setRemote(TPL, 874_289);
  });

  it('adds a community .zip from a link: download, template, on-device conversion', async () => {
    rvcInfos.clear();
    const name = (p: string) => p.split('/').pop()!;
    (fakeEngine.rvcInfo as jest.Mock).mockImplementationOnce(async (p: string) => ({
      ...defaultRvcInfo(),
      pthName: 'PeterGriffin.pth',
      sampleRate: 48000,
      info: '375epoch',
      template: 'rvc_template_v2_48k.onnx',
      _file: name(p),
    }));
    const voice = make();
    const v = await voice.addVoiceFromUrl(URL);
    expect(v.name).toBe('Peter Griffin');
    expect(v.source).toBe('url');
    expect(v.dir).toMatch(/\/voice\/voices\/v_/);
    expect(v.path).toBe(`${v.dir}/model.onnx`);
    expect(v.origin).toBe('PeterGriffin.pth · RVC v2 · 48 kHz · 375epoch');
    expect(voice.selectedVoice?.id).toBe(v.id);
    expect(voice.importing).toBeNull();
    const [src, tpl, out] = (fakeEngine.rvcImport as jest.Mock).mock.calls[0];
    expect(tpl).toBe('/docs/mobigpt/voice/base/templates/rvc_template_v2_48k.onnx');
    expect(out).toBe(v.dir);
    expect(fs.__files().has(src)).toBe(false); // the downloaded archive is cleaned up
  });

  it('downloads each template only once', async () => {
    const voice = make();
    const start = jest.spyOn(voice.downloads, 'start');
    await voice.addVoiceFromUrl(URL.replace('Peter%20Griffin', 'A'));
    fs.__setRemote(URL.replace('Peter%20Griffin', 'B'), 1000);
    await voice.addVoiceFromUrl(URL.replace('Peter%20Griffin', 'B'));
    const templateDownloads = start.mock.calls.filter(c => String(c[0]).startsWith('template:'));
    expect(templateDownloads).toHaveLength(1);
    expect((fakeEngine.rvcImport as jest.Mock).mock.calls).toHaveLength(2);
    expect(voice.voices.map(x => x.name)).toEqual(['A', 'B']);
  });

  it('explains what it cannot import', async () => {
    const voice = make();
    await expect(voice.addVoiceFromUrl('not a link')).rejects.toThrow(/https:\/\//);
    await expect(voice.addVoiceFromUrl('https://example.com/voice.mp3')).rejects.toThrow(/\.onnx, \.pth or \.zip/);
    (fakeEngine.rvcInfo as jest.Mock).mockImplementationOnce(async () => ({...defaultRvcInfo(), f0: false}));
    await expect(voice.addVoiceFromUrl(URL)).rejects.toThrow(/pitch guidance/);
    (fakeEngine.rvcInfo as jest.Mock).mockImplementationOnce(async () => ({...defaultRvcInfo(), template: 'rvc_template_v3_24k.onnx'}));
    fs.__setRemote(URL, 1000);
    await expect(voice.addVoiceFromUrl(URL)).rejects.toThrow(/not supported yet/);
    expect(voice.voices).toHaveLength(0);
    expect(voice.importing).toBeNull();
  });

  it('removing an imported voice deletes its folder', async () => {
    const voice = make();
    const v = await voice.addVoiceFromUrl(URL);
    expect(fs.__files().has(`${v.dir}/weights.bin`)).toBe(true);
    await voice.removeVoice(v.id);
    expect(voice.voices).toHaveLength(0);
    expect(fs.__files().has(`${v.dir}/weights.bin`)).toBe(false);
    expect(fs.__files().has(`${v.dir}/model.onnx`)).toBe(false);
  });

  it('lists .onnx, .pth and .zip voices in a Hub repo, but not training checkpoints or indexes', async () => {
    const voice = make();
    (voice as any).hf.listFiles = async () =>
      ['Peter Griffin.zip', 'model.pth', 'G_2333.pth', 'D_2333.pth', 'added_IVF.index', 'voice.onnx', 'README.md'].map(path => ({
        path,
        size: 1,
      }));
    expect((await voice.listVoiceFiles('x/y')).map(f => f.path)).toEqual(['Peter Griffin.zip', 'model.pth', 'voice.onnx']);
  });
});
