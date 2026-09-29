import * as FS from '@dr.pogodin/react-native-fs';

const root = `${FS.DocumentDirectoryPath}/mobigpt`;

export const Paths = {
  root,
  models: `${root}/models`,
  voices: `${root}/voice/voices`,
  voiceBase: `${root}/voice/base`,
  recordings: `${root}/voice/recordings`,
  outputs: `${root}/voice/outputs`,
  temp: `${FS.CachesDirectoryPath}/mobigpt`,
};

let ensured = false;

export async function ensureDirs(): Promise<void> {
  if (ensured) {
    return;
  }
  for (const dir of Object.values(Paths)) {
    if (!(await FS.exists(dir))) {
      await FS.mkdir(dir);
    }
  }
  ensured = true;
}

export function safeFileName(name: string): string {
  return name.replace(/[^A-Za-z0-9._-]+/g, '_').slice(0, 120) || 'file';
}

export function stripFileScheme(uri: string): string {
  return uri.startsWith('file://') ? decodeURIComponent(uri.slice(7)) : uri;
}
