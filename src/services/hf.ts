/** Minimal Hugging Face Hub client (public REST API, optional token). */
import {detectQuant, Quant, quantQuality} from '../utils/gguf';

export const HF_BASE = 'https://huggingface.co';

export interface HFModelSummary {
  id: string;
  author?: string;
  downloads: number;
  likes: number;
  lastModified?: string;
  tags: string[];
  gated?: boolean | string;
}

export interface HFFile {
  path: string;
  size: number;
  quant?: Quant;
}

type Fetch = typeof fetch;

export class HFClient {
  constructor(private token: string | undefined = undefined, private fetchImpl: Fetch = fetch) {}

  setToken(token: string | undefined) {
    this.token = token || undefined;
  }

  headers(): Record<string, string> {
    return this.token ? {Authorization: `Bearer ${this.token}`} : {};
  }

  private async json<T>(url: string): Promise<T> {
    const res = await this.fetchImpl(url, {headers: {Accept: 'application/json', ...this.headers()}});
    if (res.status === 401 || res.status === 403) {
      throw new Error('This repository is gated or private. Add a Hugging Face token in Settings.');
    }
    if (!res.ok) {
      throw new Error(`Hugging Face request failed (${res.status})`);
    }
    return (await res.json()) as T;
  }

  /** Full-text model search. `filter` narrows by library/tag, e.g. "gguf" or "onnx". */
  async search(query: string, filter: 'gguf' | 'onnx', limit = 30): Promise<HFModelSummary[]> {
    const params = new URLSearchParams({
      search: query,
      filter,
      sort: 'downloads',
      direction: '-1',
      limit: String(limit),
      full: 'false',
    });
    const raw = await this.json<any[]>(`${HF_BASE}/api/models?${params.toString()}`);
    return raw.map(m => ({
      id: m.id ?? m.modelId,
      author: m.author,
      downloads: m.downloads ?? 0,
      likes: m.likes ?? 0,
      lastModified: m.lastModified,
      tags: m.tags ?? [],
      gated: m.gated,
    }));
  }

  /** Lists files (recursively) with sizes. */
  async listFiles(repo: string, revision = 'main'): Promise<HFFile[]> {
    const raw = await this.json<any[]>(`${HF_BASE}/api/models/${encodeRepo(repo)}/tree/${encodeURIComponent(revision)}?recursive=true`);
    return raw
      .filter(e => e.type === 'file')
      .map(e => ({
        path: e.path as string,
        // LFS files report the real size under lfs.size.
        size: (e.lfs?.size ?? e.size ?? 0) as number,
        quant: (e.path as string).endsWith('.gguf') ? detectQuant(e.path) : undefined,
      }));
  }

  resolveUrl(repo: string, file: string, revision = 'main'): string {
    return `${HF_BASE}/${encodeRepo(repo)}/resolve/${encodeURIComponent(revision)}/${file.split('/').map(encodeURIComponent).join('/')}`;
  }
}

function encodeRepo(repo: string): string {
  return repo.split('/').map(encodeURIComponent).join('/');
}

/**
 * GGUF files eligible for chat: excludes mmproj/vision projectors and split
 * shards beyond the first part (llama.cpp loads split files from part 1).
 */
export function chatGgufFiles(files: HFFile[]): HFFile[] {
  return files.filter(f => {
    const name = f.path.toLowerCase();
    if (!name.endsWith('.gguf')) {
      return false;
    }
    if (name.includes('mmproj') || name.includes('projector')) {
      return false;
    }
    const split = name.match(/-(\d{5})-of-(\d{5})\.gguf$/);
    return !split || split[1] === '00001';
  });
}

/**
 * Picks the file we would recommend: the best quality quant that fits the
 * memory budget, preferring mobile-friendly Q4_K_M / Q4_0.
 */
export function recommendFile(files: HFFile[], budgetBytes: number): HFFile | undefined {
  const candidates = chatGgufFiles(files).filter(f => f.size > 0 && f.size * 1.25 < budgetBytes);
  if (!candidates.length) {
    return undefined;
  }
  const preferred = candidates.find(f => f.quant === 'Q4_K_M') ?? candidates.find(f => f.quant === 'Q4_0');
  if (preferred) {
    return preferred;
  }
  return [...candidates].sort((a, b) => quantQuality(b.quant ?? 'unknown') - quantQuality(a.quant ?? 'unknown'))[0];
}

/** Resolves a catalog file name against a live listing (case-insensitive, tolerant). */
export function matchFile(files: HFFile[], wanted: string): HFFile | undefined {
  const exact = files.find(f => f.path === wanted);
  if (exact) {
    return exact;
  }
  const lower = wanted.toLowerCase();
  return (
    files.find(f => f.path.toLowerCase() === lower) ??
    files.find(f => f.path.toLowerCase().endsWith(`/${lower}`)) ??
    files.find(f => f.path.split('/').pop()?.toLowerCase() === lower.split('/').pop())
  );
}

export function onnxFiles(files: HFFile[]): HFFile[] {
  return files.filter(f => f.path.toLowerCase().endsWith('.onnx'));
}
