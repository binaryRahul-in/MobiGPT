/**
 * Thin façade over llama.rn (llama.cpp) so the rest of the app never imports
 * native code directly — keeps stores unit-testable with a fake engine.
 */
import {getBackendDevicesInfo, initLlama, LlamaContext, loadLlamaModelInfo, releaseAllLlama} from 'llama.rn';

export type Accel = 'cpu' | 'gpu' | 'npu';

export interface LoadOptions {
  path: string;
  nCtx: number;
  nThreads: number;
  accel: Accel;
  gpuLayers: number;
  flashAttn: boolean;
  cacheType: 'f16' | 'q8_0' | 'q4_0';
  useMlock: boolean;
}

export interface LoadedModelInfo {
  description: string;
  sizeBytes: number;
  nParams: number;
  gpu: boolean;
  reasonNoGPU: string;
  devices: string[];
  systemInfo: string;
  metadata: Record<string, unknown>;
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface GenerationParams {
  temperature: number;
  topP: number;
  topK: number;
  minP: number;
  maxTokens: number;
  repeatPenalty: number;
  enableThinking?: boolean;
}

export interface GenerationResult {
  text: string;
  reasoning: string;
  tokensPredicted: number;
  tokensEvaluated: number;
  promptTps: number;
  genTps: number;
  interrupted: boolean;
  contextFull: boolean;
}

export interface BenchOutcome {
  ppTps: number;
  tgTps: number;
  nThreads: number;
  nGpuLayers: number;
  flashAttn: boolean;
}

const STOP_WORDS = [
  '</s>',
  '<|end|>',
  '<|eot_id|>',
  '<|end_of_text|>',
  '<|im_end|>',
  '<|EOT|>',
  '<|END_OF_TURN_TOKEN|>',
  '<|end_of_turn|>',
  '<end_of_turn>',
  '<|endoftext|>',
];

export interface LlmEngine {
  backendDevices(): Promise<string[]>;
  readModelInfo(path: string): Promise<Record<string, unknown>>;
  load(opts: LoadOptions, onProgress?: (p: number) => void): Promise<LoadedModelInfo>;
  unload(): Promise<void>;
  isLoaded(): boolean;
  chat(
    messages: ChatMessage[],
    params: GenerationParams,
    onToken: (t: {token: string; content?: string; reasoning?: string}) => void,
  ): Promise<GenerationResult>;
  stop(): Promise<void>;
  bench(pp: number, tg: number, pl: number, nr: number): Promise<BenchOutcome>;
}

class LlamaRnEngine implements LlmEngine {
  private ctx: LlamaContext | null = null;

  async backendDevices(): Promise<string[]> {
    try {
      const devs = await getBackendDevicesInfo();
      return devs.map(d => d.deviceName || d.backend);
    } catch {
      return [];
    }
  }

  async readModelInfo(path: string) {
    return (await loadLlamaModelInfo(path)) as Record<string, unknown>;
  }

  async load(o: LoadOptions, onProgress?: (p: number) => void): Promise<LoadedModelInfo> {
    await this.unload();
    const devices = o.accel === 'npu' ? ['HTP*'] : undefined;
    const ctx = await initLlama(
      {
        model: o.path,
        n_ctx: o.nCtx,
        n_batch: 512,
        n_ubatch: 512,
        n_threads: o.nThreads > 0 ? o.nThreads : undefined,
        n_gpu_layers: o.accel === 'cpu' ? 0 : o.gpuLayers,
        no_gpu_devices: o.accel === 'cpu',
        devices,
        flash_attn_type: o.flashAttn ? 'auto' : 'off',
        cache_type_k: o.cacheType,
        cache_type_v: o.flashAttn ? o.cacheType : 'f16',
        use_mlock: o.useMlock,
        use_mmap: true,
        use_progress_callback: !!onProgress,
      },
      onProgress ? (p: number) => onProgress(Math.min(1, p / 100)) : undefined,
    );
    this.ctx = ctx;
    return {
      description: ctx.model.desc,
      sizeBytes: ctx.model.size,
      nParams: ctx.model.nParams,
      gpu: ctx.gpu,
      reasonNoGPU: ctx.reasonNoGPU,
      devices: ctx.devices ?? [],
      systemInfo: (ctx as any).systemInfo ?? '',
      metadata: (ctx.model.metadata ?? {}) as Record<string, unknown>,
    };
  }

  async unload(): Promise<void> {
    const c = this.ctx;
    this.ctx = null;
    if (c) {
      await c.release();
    } else {
      await releaseAllLlama().catch(() => undefined);
    }
  }

  isLoaded(): boolean {
    return this.ctx != null;
  }

  async chat(messages: ChatMessage[], p: GenerationParams, onToken: (t: {token: string; content?: string; reasoning?: string}) => void) {
    if (!this.ctx) {
      throw new Error('No model loaded');
    }
    const res = await this.ctx.completion(
      {
        messages,
        n_predict: p.maxTokens,
        temperature: p.temperature,
        top_p: p.topP,
        top_k: p.topK,
        min_p: p.minP,
        penalty_repeat: p.repeatPenalty,
        stop: STOP_WORDS,
        jinja: true,
        enable_thinking: p.enableThinking ?? false,
        reasoning_format: 'auto',
      } as any,
      data => onToken({token: data.token, content: data.content, reasoning: data.reasoning_content}),
    );
    return {
      text: res.content || res.text,
      reasoning: res.reasoning_content ?? '',
      tokensPredicted: res.tokens_predicted,
      tokensEvaluated: res.tokens_evaluated,
      promptTps: res.timings?.prompt_per_second ?? 0,
      genTps: res.timings?.predicted_per_second ?? 0,
      interrupted: res.interrupted,
      contextFull: res.context_full,
    };
  }

  async stop() {
    await this.ctx?.stopCompletion();
  }

  async bench(pp: number, tg: number, pl: number, nr: number): Promise<BenchOutcome> {
    if (!this.ctx) {
      throw new Error('Load a model first');
    }
    const r = await this.ctx.bench(pp, tg, pl, nr);
    return {ppTps: r.speedPp, tgTps: r.speedTg, nThreads: r.nThreads, nGpuLayers: r.nGpuLayers, flashAttn: !!r.flashAttn};
  }
}

let engine: LlmEngine | null = null;

export function getLlmEngine(): LlmEngine {
  if (!engine) {
    engine = new LlamaRnEngine();
  }
  return engine;
}

/** Test hook. */
export function setLlmEngine(e: LlmEngine | null) {
  engine = e;
}
