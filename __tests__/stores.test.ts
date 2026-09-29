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
    inspections.set('/docs/mobigpt/voice/voices/voice.onnx', {
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
