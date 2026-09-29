import * as FS from '@dr.pogodin/react-native-fs';
import {action, makeObservable, observable, runInAction} from 'mobx';

export type DownloadState = 'queued' | 'downloading' | 'done' | 'error' | 'cancelled';

export interface DownloadTask {
  id: string;
  url: string;
  dest: string;
  label: string;
  bytes: number;
  total: number;
  speedBps: number;
  state: DownloadState;
  error?: string;
}

/** A download that receives no bytes for this long is aborted instead of hanging forever. */
export const STALL_TIMEOUT_MS = 45_000;

/**
 * Downloads to `<dest>.part` and renames on success, so a half-written file
 * is never mistaken for a model. Progress is observable (mobx).
 */
export class DownloadManager {
  tasks = new Map<string, DownloadTask>();
  private jobs = new Map<string, number>();

  constructor(private headers: () => Record<string, string> = () => ({})) {
    makeObservable(this, {tasks: observable, start: action, cancel: action});
  }

  get(id: string): DownloadTask | undefined {
    return this.tasks.get(id);
  }

  isActive(id: string): boolean {
    const t = this.tasks.get(id);
    return !!t && (t.state === 'queued' || t.state === 'downloading');
  }

  async start(id: string, url: string, dest: string, label: string, expectedBytes = 0): Promise<string> {
    if (this.isActive(id)) {
      throw new Error('Download already in progress');
    }
    const task: DownloadTask = {id, url, dest, label, bytes: 0, total: expectedBytes, speedBps: 0, state: 'queued'};
    this.tasks.set(id, task);
    const part = `${dest}.part`;
    if (await FS.exists(part)) {
      await FS.unlink(part);
    }
    const dir = dest.substring(0, dest.lastIndexOf('/'));
    if (dir && !(await FS.exists(dir))) {
      await FS.mkdir(dir);
    }
    let lastT = Date.now();
    let lastB = 0;
    let lastActivity = Date.now();
    let stalled = false;
    const {jobId, promise} = FS.downloadFile({
      fromUrl: url,
      toFile: part,
      headers: {'User-Agent': 'MobiGPT/1.0 (+https://github.com/binaryRahul-in/MobiGPT)', ...this.headers()},
      connectionTimeout: 20_000,
      readTimeout: 30_000,
      progressInterval: 500,
      background: true,
      discretionary: false,
      begin: res =>
        runInAction(() => {
          lastActivity = Date.now();
          console.log(`[download] ${label}: HTTP ${res.statusCode}, ${res.contentLength} bytes from ${url}`);
          const t = this.tasks.get(id);
          if (t) {
            t.state = 'downloading';
            t.total = res.contentLength > 0 ? res.contentLength : t.total;
          }
        }),
      progress: res =>
        runInAction(() => {
          const t = this.tasks.get(id);
          if (!t) {
            return;
          }
          const now = Date.now();
          if (res.bytesWritten > lastB) {
            lastActivity = now;
          }
          if (now - lastT >= 500) {
            t.speedBps = ((res.bytesWritten - lastB) * 1000) / (now - lastT);
            lastT = now;
            lastB = res.bytesWritten;
          }
          t.bytes = res.bytesWritten;
          if (res.contentLength > 0) {
            t.total = res.contentLength;
          }
        }),
    });
    this.jobs.set(id, jobId);
    const watchdog = setInterval(() => {
      if (Date.now() - lastActivity > STALL_TIMEOUT_MS) {
        stalled = true;
        FS.stopDownload(jobId);
      }
    }, 5_000);
    try {
      const res = await promise;
      if (stalled) {
        throw new Error('stalled');
      }
      if (this.tasks.get(id)?.state === 'cancelled') {
        throw new Error('cancelled');
      }
      if (res.statusCode < 200 || res.statusCode >= 300) {
        throw new Error(
          res.statusCode === 401 || res.statusCode === 403
            ? 'Access denied: this repository is gated. Add your Hugging Face token in Settings.'
            : `Server responded ${res.statusCode}`,
        );
      }
      const stat = await FS.stat(part);
      if (expectedBytes > 0 && Math.abs(stat.size - expectedBytes) > Math.max(1024, expectedBytes * 0.02)) {
        throw new Error(`Incomplete download (${stat.size} of ${expectedBytes} bytes)`);
      }
      if (await FS.exists(dest)) {
        await FS.unlink(dest);
      }
      await FS.moveFile(part, dest);
      runInAction(() => {
        const t = this.tasks.get(id);
        if (t) {
          t.state = 'done';
          t.bytes = stat.size;
          t.total = stat.size;
        }
      });
      console.log(`[download] ${label}: done, ${stat.size} bytes`);
      return dest;
    } catch (err: any) {
      const cancelled = !stalled && this.tasks.get(id)?.state === 'cancelled';
      const e = stalled
        ? new Error(`${label}: no data received for ${STALL_TIMEOUT_MS / 1000} s. Check your connection and try again.`)
        : err;
      console.warn(`[download] ${label} failed: ${e?.message ?? e} (${url})`);
      runInAction(() => {
        const t = this.tasks.get(id);
        if (t && !cancelled) {
          t.state = 'error';
          t.error = e?.message ?? String(e);
        }
      });
      FS.exists(part)
        .then(x => (x ? FS.unlink(part) : undefined))
        .catch(() => undefined);
      throw cancelled ? new Error('Download cancelled') : e;
    } finally {
      clearInterval(watchdog);
      this.jobs.delete(id);
    }
  }

  cancel(id: string) {
    const job = this.jobs.get(id);
    const t = this.tasks.get(id);
    if (t) {
      t.state = 'cancelled';
    }
    if (job != null) {
      FS.stopDownload(job);
    }
  }

  clear(id: string) {
    runInAction(() => this.tasks.delete(id));
  }
}
