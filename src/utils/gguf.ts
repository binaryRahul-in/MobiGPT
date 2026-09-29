/**
 * GGUF helpers: quantisation detection and memory estimation.
 * The estimator follows llama.cpp's allocation model (weights + KV cache +
 * compute buffers) and is adapted from PocketPal AI (MIT) — see NOTICE.md.
 */

export type Quant =
  | 'F32'
  | 'F16'
  | 'BF16'
  | 'Q8_0'
  | 'Q6_K'
  | 'Q5_K_M'
  | 'Q5_K_S'
  | 'Q5_0'
  | 'Q4_K_M'
  | 'Q4_K_S'
  | 'Q4_0'
  | 'IQ4_XS'
  | 'IQ4_NL'
  | 'Q3_K_L'
  | 'Q3_K_M'
  | 'Q3_K_S'
  | 'IQ3_M'
  | 'IQ3_XXS'
  | 'Q2_K'
  | 'IQ2_M'
  | 'IQ2_XXS'
  | 'IQ1_S'
  | 'unknown';

const QUANT_PATTERNS: Array<[RegExp, Quant]> = [
  [/(?:^|[-_.])bf16(?:[-_.]|$)/i, 'BF16'],
  [/(?:^|[-_.])f32(?:[-_.]|$)/i, 'F32'],
  [/(?:^|[-_.])f16(?:[-_.]|$)/i, 'F16'],
  [/q8_0/i, 'Q8_0'],
  [/q6_k/i, 'Q6_K'],
  [/q5_k_m/i, 'Q5_K_M'],
  [/q5_k_s/i, 'Q5_K_S'],
  [/q5_0/i, 'Q5_0'],
  [/q4_k_m/i, 'Q4_K_M'],
  [/q4_k_s/i, 'Q4_K_S'],
  [/iq4_xs/i, 'IQ4_XS'],
  [/iq4_nl/i, 'IQ4_NL'],
  [/q4_0/i, 'Q4_0'],
  [/q3_k_l/i, 'Q3_K_L'],
  [/q3_k_m/i, 'Q3_K_M'],
  [/q3_k_s/i, 'Q3_K_S'],
  [/iq3_m/i, 'IQ3_M'],
  [/iq3_xxs/i, 'IQ3_XXS'],
  [/q2_k/i, 'Q2_K'],
  [/iq2_m/i, 'IQ2_M'],
  [/iq2_xxs/i, 'IQ2_XXS'],
  [/iq1_s/i, 'IQ1_S'],
];

export function detectQuant(filename: string): Quant {
  for (const [re, q] of QUANT_PATTERNS) {
    if (re.test(filename)) {
      return q;
    }
  }
  return 'unknown';
}

/** Rough quality rank (higher is better) used to recommend a file. */
export function quantQuality(q: Quant): number {
  const order: Quant[] = [
    'IQ1_S',
    'IQ2_XXS',
    'IQ2_M',
    'Q2_K',
    'IQ3_XXS',
    'IQ3_M',
    'Q3_K_S',
    'Q3_K_M',
    'Q3_K_L',
    'Q4_0',
    'IQ4_NL',
    'IQ4_XS',
    'Q4_K_S',
    'Q4_K_M',
    'Q5_0',
    'Q5_K_S',
    'Q5_K_M',
    'Q6_K',
    'Q8_0',
    'BF16',
    'F16',
    'F32',
  ];
  const i = order.indexOf(q);
  return i < 0 ? 0 : i + 1;
}

/** ARM-repacked quants are fastest on phones with dotprod/i8mm. */
export function isMobileFriendlyQuant(q: Quant): boolean {
  return ['Q4_0', 'Q4_K_M', 'Q4_K_S', 'IQ4_NL', 'IQ4_XS', 'Q3_K_M', 'Q5_K_M', 'Q8_0'].includes(q);
}

export interface GGUFMetadata {
  nLayers: number;
  nEmbd: number;
  nHead: number;
  nHeadKv: number;
  nVocab: number;
  headDimK?: number;
  headDimV?: number;
  slidingWindow?: number;
  architecture?: string;
  contextLength?: number;
}

export interface MemoryEstimateInput {
  fileSizeBytes: number;
  nCtx: number;
  nUbatch?: number;
  cacheTypeK?: string;
  cacheTypeV?: string;
  meta?: GGUFMetadata;
}

const KV_BYTES: Record<string, number> = {
  f32: 4,
  f16: 2,
  bf16: 2,
  q8_0: 34 / 32,
  q4_0: 18 / 32,
  q4_1: 20 / 32,
  q5_0: 22 / 32,
  q5_1: 24 / 32,
};

export function kvBytesPerElement(type = 'f16'): number {
  return KV_BYTES[type.toLowerCase()] ?? 2;
}

/**
 * Estimated peak RAM for running a model: weights (mmap'd, but resident once
 * warmed up) + KV cache + compute buffers, with 10 % runtime overhead. Without
 * GGUF metadata falls back to 1.2 x file size + a context-proportional term.
 */
export function estimateModelMemory(i: MemoryEstimateInput): number {
  const nUbatch = i.nUbatch ?? 512;
  const m = i.meta;
  if (m && m.nLayers > 0 && m.nHeadKv > 0 && m.nEmbd > 0) {
    const headK = m.headDimK ?? m.nEmbd / Math.max(1, m.nHead);
    const headV = m.headDimV ?? headK;
    const ctx = m.slidingWindow ? Math.min(i.nCtx, m.slidingWindow) : i.nCtx;
    const kv = m.nLayers * ctx * m.nHeadKv * (headK * kvBytesPerElement(i.cacheTypeK) + headV * kvBytesPerElement(i.cacheTypeV));
    const compute = (m.nVocab + m.nEmbd) * nUbatch * 4;
    return (i.fileSizeBytes + kv + compute) * 1.1;
  }
  // ~0.1 MB of KV per token for 1-3B models at f16 is a conservative default.
  return i.fileSizeBytes * 1.2 + i.nCtx * 0.1e6;
}

/** Parses the model-info object returned by llama.rn's loadLlamaModelInfo. */
export function parseLlamaModelInfo(info: Record<string, unknown>): GGUFMetadata | undefined {
  const arch = String(info['general.architecture'] ?? '');
  if (!arch) {
    return undefined;
  }
  const num = (k: string) => {
    const v = info[`${arch}.${k}`];
    const n = typeof v === 'string' ? parseFloat(v) : (v as number);
    return typeof n === 'number' && isFinite(n) ? n : undefined;
  };
  const nEmbd = num('embedding_length');
  const nHead = num('attention.head_count');
  const nLayers = num('block_count');
  if (!nEmbd || !nHead || !nLayers) {
    return undefined;
  }
  const tokens = info['tokenizer.ggml.tokens'];
  const nVocab = num('vocab_size') ?? (Array.isArray(tokens) ? tokens.length : 32000);
  return {
    architecture: arch,
    nLayers,
    nEmbd,
    nHead,
    nHeadKv: num('attention.head_count_kv') ?? nHead,
    nVocab,
    headDimK: num('attention.key_length'),
    headDimV: num('attention.value_length'),
    slidingWindow: num('attention.sliding_window'),
    contextLength: num('context_length'),
  };
}
