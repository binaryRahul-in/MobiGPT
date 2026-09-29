/** In-memory stand-in for @dr.pogodin/react-native-fs. */
export const DocumentDirectoryPath = '/docs';
export const CachesDirectoryPath = '/caches';
export const TemporaryDirectoryPath = '/tmp';

const files = new Map<string, number>();
const dirs = new Set<string>(['/docs', '/caches', '/tmp']);
const remotes = new Map<string, {size: number; status: number}>();
const nameSizes = new Map<string, number>();
let nextJob = 1;
const cancelled = new Set<number>();

export function __reset() {
  files.clear();
  remotes.clear();
  nameSizes.clear();
  cancelled.clear();
  dirs.clear();
  ['/docs', '/caches', '/tmp'].forEach(d => dirs.add(d));
}
export function __putFile(path: string, size: number) {
  files.set(path, size);
}
/** Default size served for any URL ending in `/<name>`. */
export function __setRemoteName(name: string, size: number) {
  nameSizes.set(name, size);
}
export function __setRemote(url: string, size: number, status = 200) {
  remotes.set(url, {size, status});
}
export function __files() {
  return new Map(files);
}

export async function exists(p: string) {
  return files.has(p) || dirs.has(p);
}
export async function mkdir(p: string) {
  dirs.add(p);
}
export async function unlink(p: string) {
  if (!files.delete(p) && !dirs.delete(p)) {
    throw new Error(`ENOENT ${p}`);
  }
}
export async function moveFile(a: string, b: string) {
  const s = files.get(a);
  if (s == null) {
    throw new Error(`ENOENT ${a}`);
  }
  files.delete(a);
  files.set(b, s);
}
export async function stat(p: string) {
  const s = files.get(p);
  if (s == null) {
    throw new Error(`ENOENT ${p}`);
  }
  return {path: p, size: s, mode: 0, ctime: new Date(), mtime: new Date(), isFile: () => true, isDirectory: () => false};
}
export async function getFSInfo() {
  return {totalSpace: 128e9, freeSpace: 64e9, totalSpaceEx: 128e9, freeSpaceEx: 64e9};
}
export function stopDownload(jobId: number) {
  cancelled.add(jobId);
}
export function downloadFile(opts: {
  fromUrl: string;
  toFile: string;
  begin?: (r: {contentLength: number; jobId: number; statusCode: number; headers: object}) => void;
  progress?: (r: {contentLength: number; bytesWritten: number; jobId: number}) => void;
}) {
  const jobId = nextJob++;
  const byName = nameSizes.get(opts.fromUrl.split('/').pop() ?? '');
  const remote = remotes.get(opts.fromUrl) ?? {size: byName ?? 1000, status: 200};
  const promise = (async () => {
    await Promise.resolve();
    opts.begin?.({contentLength: remote.size, jobId, statusCode: remote.status, headers: {}});
    for (let i = 1; i <= 4; i++) {
      await new Promise(r => setTimeout(r, 1));
      if (cancelled.has(jobId)) {
        throw new Error('Download has been aborted');
      }
      opts.progress?.({contentLength: remote.size, bytesWritten: (remote.size * i) / 4, jobId});
    }
    if (remote.status === 200) {
      files.set(opts.toFile, remote.size);
    }
    return {jobId, statusCode: remote.status, bytesWritten: remote.size};
  })();
  return {jobId, promise};
}
