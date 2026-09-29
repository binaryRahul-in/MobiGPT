import {compareVersions} from '../utils/format';

export const REPO = 'binaryRahul-in/MobiGPT';
// HEAD resolves to the default branch, whatever it is named.
export const RAW_BASE = `https://raw.githubusercontent.com/${REPO}/HEAD`;

export interface ReleaseInfo {
  version: string;
  name: string;
  notes: string;
  url: string;
  publishedAt: string;
  apkUrl?: string;
}

export interface UpdateCheck {
  current: string;
  latest?: ReleaseInfo;
  updateAvailable: boolean;
}

/** Checks GitHub Releases for a newer app build. */
export async function checkForAppUpdate(current: string, fetchImpl: typeof fetch = fetch): Promise<UpdateCheck> {
  const res = await fetchImpl(`https://api.github.com/repos/${REPO}/releases/latest`, {
    headers: {Accept: 'application/vnd.github+json'},
  });
  if (res.status === 404) {
    return {current, updateAvailable: false};
  }
  if (!res.ok) {
    throw new Error(`GitHub responded ${res.status}`);
  }
  const r: any = await res.json();
  const latest: ReleaseInfo = {
    version: String(r.tag_name ?? '').replace(/^v/, ''),
    name: r.name ?? r.tag_name,
    notes: r.body ?? '',
    url: r.html_url,
    publishedAt: r.published_at,
    apkUrl: (r.assets ?? []).find((a: any) => String(a.name).endsWith('.apk'))?.browser_download_url,
  };
  return {current, latest, updateAvailable: !!latest.version && compareVersions(latest.version, current) > 0};
}

/**
 * Fetches the live model/voice catalog so new models can ship without an app
 * update. Falls back to the bundled copy when offline or on schema mismatch.
 */
export async function fetchCatalog<T extends {version: number}>(
  name: 'models' | 'voices',
  bundled: T,
  fetchImpl: typeof fetch = fetch,
): Promise<{catalog: T; source: 'remote' | 'bundled'}> {
  try {
    const res = await fetchImpl(`${RAW_BASE}/catalog/${name}.json`, {headers: {'Cache-Control': 'no-cache'}});
    if (!res.ok) {
      return {catalog: bundled, source: 'bundled'};
    }
    const remote = (await res.json()) as T;
    if (remote?.version !== bundled.version) {
      return {catalog: bundled, source: 'bundled'};
    }
    return {catalog: remote, source: 'remote'};
  } catch {
    return {catalog: bundled, source: 'bundled'};
  }
}
