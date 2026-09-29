const UNITS = ['B', 'KB', 'MB', 'GB', 'TB'];

/** Human readable byte size using decimal (SI) units, like app stores do. */
export function formatBytes(bytes: number | undefined | null, digits = 1): string {
  if (bytes == null || !isFinite(bytes) || bytes < 0) {
    return '—';
  }
  let v = bytes;
  let i = 0;
  while (v >= 1000 && i < UNITS.length - 1) {
    v /= 1000;
    i++;
  }
  return `${v.toFixed(i === 0 ? 0 : digits)} ${UNITS[i]}`;
}

export function formatGB(bytes: number, digits = 1): string {
  return `${(bytes / 1e9).toFixed(digits)} GB`;
}

export function formatDuration(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) {
    return '—';
  }
  if (seconds < 60) {
    return `${seconds.toFixed(seconds < 10 ? 1 : 0)}s`;
  }
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  if (m < 60) {
    return `${m}m ${s.toString().padStart(2, '0')}s`;
  }
  return `${Math.floor(m / 60)}h ${(m % 60).toString().padStart(2, '0')}m`;
}

export function formatRate(value: number, unit: string, digits = 1): string {
  return isFinite(value) ? `${value.toFixed(digits)} ${unit}` : '—';
}

export function formatPercent(fraction: number): string {
  return `${Math.round(Math.max(0, Math.min(1, fraction)) * 100)}%`;
}

/** Semantic-version comparison: returns >0 if a is newer than b. */
export function compareVersions(a: string, b: string): number {
  const pa = a.replace(/^v/, '').split(/[.-]/);
  const pb = b.replace(/^v/, '').split(/[.-]/);
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    const x = parseInt(pa[i] ?? '0', 10);
    const y = parseInt(pb[i] ?? '0', 10);
    if (isNaN(x) || isNaN(y)) {
      const c = (pa[i] ?? '').localeCompare(pb[i] ?? '');
      if (c !== 0) {
        return c;
      }
      continue;
    }
    if (x !== y) {
      return x - y;
    }
  }
  return 0;
}

export function uid(prefix = ''): string {
  return `${prefix}${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
}
