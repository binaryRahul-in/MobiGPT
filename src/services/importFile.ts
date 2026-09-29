import * as FS from '@dr.pogodin/react-native-fs';
import {errorCodes, isErrorWithCode, keepLocalCopy, pick, types} from '@react-native-documents/picker';

import {safeFileName, stripFileScheme} from './paths';

export interface ImportedFile {
  path: string;
  name: string;
  size: number;
}

/**
 * Lets the user pick a file (GGUF model, ONNX voice, audio clip), copies it
 * into app storage and returns the local path. Returns null if cancelled.
 */
export async function pickAndImport(destDir: string, extensions: string[], kind: 'any' | 'audio' = 'any'): Promise<ImportedFile | null> {
  let picked;
  try {
    [picked] = await pick({type: [kind === 'audio' ? types.audio : types.allFiles], allowMultiSelection: false});
  } catch (e) {
    if (isErrorWithCode(e) && e.code === errorCodes.OPERATION_CANCELED) {
      return null;
    }
    throw e;
  }
  const name = safeFileName(picked.name ?? `import-${Date.now()}`);
  const lower = name.toLowerCase();
  if (extensions.length && !extensions.some(ext => lower.endsWith(ext))) {
    throw new Error(`Unsupported file type. Expected ${extensions.join(', ')}`);
  }
  const [copy] = await keepLocalCopy({files: [{uri: picked.uri, fileName: name}], destination: 'documentDirectory'});
  if (copy.status !== 'success') {
    throw new Error(copy.copyError || 'Could not copy the file into app storage');
  }
  const src = stripFileScheme(copy.localUri);
  let dest = `${destDir}/${name}`;
  if (await FS.exists(dest)) {
    const dot = name.lastIndexOf('.');
    dest = `${destDir}/${dot > 0 ? name.slice(0, dot) : name}-${Date.now()}${dot > 0 ? name.slice(dot) : ''}`;
  }
  await FS.moveFile(src, dest);
  const stat = await FS.stat(dest);
  return {path: dest, name, size: stat.size};
}
