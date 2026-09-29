import {deviceTier, gpuOffloadSupport, hexagonVersion, npuSupport} from '../src/features/device';
import {FEATURES, recommendedVoiceSettings} from '../src/features/registry';
import {evaluate, memoryFit} from '../src/features/requirements';
import {chatGgufFiles, HFClient, HFFile, matchFile, recommendFile} from '../src/services/hf';
import {checkForAppUpdate, fetchCatalog} from '../src/services/updates';
import {profile} from '../jest/fixtures';
import {compareVersions, formatBytes, formatDuration} from '../src/utils/format';
import {detectQuant, estimateModelMemory, parseLlamaModelInfo, quantQuality} from '../src/utils/gguf';

describe('format', () => {
  it('formats bytes, durations and versions', () => {
    expect(formatBytes(0)).toBe('0 B');
    expect(formatBytes(1_500_000)).toBe('1.5 MB');
    expect(formatBytes(2.49e9)).toBe('2.5 GB');
    expect(formatBytes(undefined)).toBe('—');
    expect(formatDuration(5.25)).toBe('5.3s');
    expect(formatDuration(125)).toBe('2m 05s');
    expect(compareVersions('1.2.0', '1.10.0')).toBeLessThan(0);
    expect(compareVersions('v2.0.0', '1.9.9')).toBeGreaterThan(0);
    expect(compareVersions('1.0.0', '1.0')).toBe(0);
  });
});

describe('gguf', () => {
  it('detects quantisation from file names', () => {
    expect(detectQuant('Qwen3-0.6B-Q4_K_M.gguf')).toBe('Q4_K_M');
    expect(detectQuant('model-q8_0.gguf')).toBe('Q8_0');
    expect(detectQuant('gemma-3-1b-it-BF16.gguf')).toBe('BF16');
    expect(detectQuant('llama-IQ4_XS.gguf')).toBe('IQ4_XS');
    expect(detectQuant('weird.gguf')).toBe('unknown');
    expect(quantQuality('Q8_0')).toBeGreaterThan(quantQuality('Q4_K_M'));
  });

  it('estimates memory with and without metadata', () => {
    const meta = {nLayers: 16, nEmbd: 2048, nHead: 32, nHeadKv: 8, nVocab: 128256};
    const small = estimateModelMemory({fileSizeBytes: 800e6, nCtx: 2048, meta});
    const big = estimateModelMemory({fileSizeBytes: 800e6, nCtx: 8192, meta});
    expect(small).toBeGreaterThan(800e6);
    expect(big).toBeGreaterThan(small);
    // q8_0 KV halves the cache vs f16
    const q8 = estimateModelMemory({fileSizeBytes: 800e6, nCtx: 8192, meta, cacheTypeK: 'q8_0', cacheTypeV: 'q8_0'});
    expect(q8).toBeLessThan(big);
    // sliding window caps KV
    const swa = estimateModelMemory({fileSizeBytes: 800e6, nCtx: 8192, meta: {...meta, slidingWindow: 512}});
    expect(swa).toBeLessThan(big);
    expect(estimateModelMemory({fileSizeBytes: 1e9, nCtx: 2048})).toBeCloseTo(1.2e9 + 2048 * 0.1e6);
  });

  it('parses llama.rn model info', () => {
    const meta = parseLlamaModelInfo({
      'general.architecture': 'llama',
      'llama.embedding_length': 2048,
      'llama.attention.head_count': 32,
      'llama.attention.head_count_kv': '8',
      'llama.block_count': 16,
      'llama.context_length': 131072,
      'llama.vocab_size': 128256,
    });
    expect(meta).toMatchObject({nLayers: 16, nHeadKv: 8, nVocab: 128256, contextLength: 131072});
    expect(parseLlamaModelInfo({})).toBeUndefined();
  });
});

describe('hugging face helpers', () => {
  const files: HFFile[] = [
    {path: 'model-Q2_K.gguf', size: 0.6e9, quant: 'Q2_K'},
    {path: 'model-Q4_K_M.gguf', size: 1.1e9, quant: 'Q4_K_M'},
    {path: 'model-Q8_0.gguf', size: 2.0e9, quant: 'Q8_0'},
    {path: 'mmproj-model-f16.gguf', size: 0.4e9},
    {path: 'big-Q4_K_M-00002-of-00003.gguf', size: 5e9},
    {path: 'README.md', size: 10},
  ];

  it('filters chat GGUF files and recommends one that fits', () => {
    expect(chatGgufFiles(files).map(f => f.path)).toEqual(['model-Q2_K.gguf', 'model-Q4_K_M.gguf', 'model-Q8_0.gguf']);
    expect(recommendFile(files, 3e9)?.path).toBe('model-Q4_K_M.gguf');
    expect(recommendFile(files, 1.0e9)?.path).toBe('model-Q2_K.gguf');
    expect(recommendFile(files, 0.1e9)).toBeUndefined();
  });

  it('matches catalog files tolerantly', () => {
    const list: HFFile[] = [
      {path: 'onnx/fcpe.onnx', size: 1},
      {path: 'Qwen3-0.6B-Q4_K_M.gguf', size: 2},
    ];
    expect(matchFile(list, 'onnx/fcpe.onnx')?.size).toBe(1);
    expect(matchFile(list, 'fcpe.onnx')?.size).toBe(1);
    expect(matchFile(list, 'qwen3-0.6b-q4_k_m.gguf')?.size).toBe(2);
    expect(matchFile(list, 'missing.onnx')).toBeUndefined();
  });

  it('lists files and resolves URLs via the REST API', async () => {
    const fetchMock = jest.fn(async (url: string) => ({
      ok: true,
      status: 200,
      json: async () =>
        url.includes('/tree/')
          ? [
              {type: 'file', path: 'a-Q4_K_M.gguf', size: 100, lfs: {size: 1234}},
              {type: 'directory', path: 'sub'},
            ]
          : [{id: 'org/repo', downloads: 5, likes: 1, tags: ['gguf']}],
    }));
    const hf = new HFClient('tok', fetchMock as any);
    const listing = await hf.listFiles('org/repo');
    expect(listing).toEqual([{path: 'a-Q4_K_M.gguf', size: 1234, quant: 'Q4_K_M'}]);
    expect((fetchMock.mock.calls[0] as any)[1].headers.Authorization).toBe('Bearer tok');
    const res = await hf.search('qwen', 'gguf');
    expect(res[0].id).toBe('org/repo');
    expect(hf.resolveUrl('org/repo', 'dir/file name.onnx')).toBe('https://huggingface.co/org/repo/resolve/main/dir/file%20name.onnx');
  });

  it('explains gated repositories', async () => {
    const hf = new HFClient(undefined, (async () => ({ok: false, status: 401, json: async () => ({})})) as any);
    await expect(hf.listFiles('meta/llama')).rejects.toThrow(/gated/);
  });
});

describe('device capability detection', () => {
  it('classifies tiers', () => {
    expect(deviceTier({totalRam: 2e9, cores: 4})).toBe('entry');
    expect(deviceTier({totalRam: 4e9, cores: 8})).toBe('low');
    expect(deviceTier({totalRam: 6e9, cores: 8})).toBe('mid');
    expect(deviceTier({totalRam: 8e9, cores: 8})).toBe('high');
    expect(deviceTier({totalRam: 12e9, cores: 8})).toBe('flagship');
  });

  it('gates GPU offload on Adreno + dotprod + i8mm', () => {
    expect(gpuOffloadSupport(profile()).supported).toBe(true);
    expect(gpuOffloadSupport(profile({gpu: {name: 'Mali', type: 'Mali', adreno: false, mali: true, apple: false}})).supported).toBe(false);
    expect(gpuOffloadSupport(profile({hasI8mm: false})).reason).toMatch(/i8mm/);
    expect(gpuOffloadSupport(profile({isEmulator: true})).supported).toBe(false);
  });

  it('maps Snapdragon SoCs to Hexagon generations', () => {
    expect(hexagonVersion('SM8650')).toBe('v75');
    expect(hexagonVersion('Snapdragon 8 Elite')).toBe('v79');
    expect(hexagonVersion('SM8450')).toBe('v69');
    expect(hexagonVersion('Dimensity 9300')).toBeNull();
    expect(npuSupport(profile()).supported).toBe(true);
    expect(npuSupport(profile({soc: 'SM8450'})).experimental).toBe(true);
    expect(npuSupport(profile({hasNpu: false, soc: 'Exynos'})).supported).toBe(false);
    expect(npuSupport(profile({platform: 'ios'})).supported).toBe(false);
  });
});

describe('requirements engine', () => {
  const voice = FEATURES.find(f => f.id === 'voice')!;
  const live = FEATURES.find(f => f.id === 'liveVoice')!;

  it('passes on a flagship', () => {
    expect(evaluate(voice.requirement, profile({totalRam: 12e9})).status).toBe('ok');
  });

  it('warns on 4 GB phones and blocks tiny ones', () => {
    const ev = evaluate(voice.requirement, profile({totalRam: 4e9}));
    expect(ev.status).toBe('warn');
    expect(ev.issues[0].code).toBe('ram');
    expect(evaluate(voice.requirement, profile({totalRam: 2e9})).status).toBe('block');
  });

  it('blocks when the native module is compiled out or storage is full', () => {
    expect(evaluate(voice.requirement, profile({voiceModuleAvailable: false})).issues.map(i => i.code)).toContain('native');
    expect(evaluate(voice.requirement, profile({freeStorage: 50e6})).status).toBe('block');
  });

  it('warns about slow CPUs for live voice', () => {
    const ev = evaluate(live.requirement, profile({maxFreqMhz: 2000, cores: 6, totalRam: 6e9}));
    expect(ev.status).toBe('warn');
    expect(ev.issues.map(i => i.code)).toEqual(expect.arrayContaining(['ram', 'cpu', 'freq']));
  });

  it('checks memory fit against realistic free RAM', () => {
    expect(memoryFit(1e9, profile()).severity).toBe('ok');
    expect(memoryFit(4.5e9, profile()).severity).toBe('warn');
    expect(memoryFit(7.5e9, profile()).severity).toBe('block');
  });

  it('every feature option is well formed', () => {
    for (const f of FEATURES) {
      for (const o of f.options ?? []) {
        if (o.type === 'choice') {
          expect(o.choices?.length).toBeGreaterThan(1);
        }
        if (o.type === 'locked') {
          expect(o.lockedReason).toBeTruthy();
        }
      }
    }
    const keys = voice.options!.map(o => o.key);
    expect(keys).toEqual(expect.arrayContaining(['pitchMethod', 'indexRate', 'chunkSeconds', 'nativePath', 'encoderPrecision']));
  });

  it('recommends lighter voice settings on small phones', () => {
    expect(recommendedVoiceSettings(profile({totalRam: 3e9}))).toMatchObject({
      pitchMethod: 'dio',
      loadStrategy: 'sequential',
      encoderPrecision: 'int8',
    });
    expect(recommendedVoiceSettings(profile({totalRam: 4.5e9})).pitchMethod).toBe('fcpe');
    expect(recommendedVoiceSettings(profile({totalRam: 16e9}))).toMatchObject({pitchMethod: 'rmvpe', encoderPrecision: 'fp32'});
    const s = recommendedVoiceSettings(profile());
    expect(s.chunkSeconds).toBeGreaterThanOrEqual(2);
    expect(s.chunkSeconds).toBeLessThanOrEqual(3);
  });
});

describe('updates', () => {
  it('detects newer releases', async () => {
    const f = jest.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => ({
        tag_name: 'v1.2.0',
        name: 'MobiGPT 1.2.0',
        body: 'notes',
        html_url: 'u',
        assets: [{name: 'mobigpt.apk', browser_download_url: 'apk'}],
      }),
    }));
    const r = await checkForAppUpdate('1.0.0', f as any);
    expect(r.updateAvailable).toBe(true);
    expect(r.latest?.apkUrl).toBe('apk');
    expect((await checkForAppUpdate('1.2.0', f as any)).updateAvailable).toBe(false);
    const none = await checkForAppUpdate('1.0.0', (async () => ({ok: false, status: 404})) as any);
    expect(none.updateAvailable).toBe(false);
  });

  it('falls back to the bundled catalog', async () => {
    const bundled = {version: 1, models: []};
    expect(
      (
        await fetchCatalog('models', bundled, (async () => {
          throw new Error('offline');
        }) as any)
      ).source,
    ).toBe('bundled');
    expect((await fetchCatalog('models', bundled, (async () => ({ok: true, json: async () => ({version: 2})})) as any)).source).toBe(
      'bundled',
    );
    expect(
      (await fetchCatalog('models', bundled, (async () => ({ok: true, json: async () => ({version: 1, models: [1]})})) as any)).source,
    ).toBe('remote');
  });
});
