import mockAsyncStorage from '@react-native-async-storage/async-storage/jest/async-storage-mock';
import mockSafeAreaContext from 'react-native-safe-area-context/jest/mock';

jest.mock('@react-native-async-storage/async-storage', () => mockAsyncStorage);
jest.mock('react-native-safe-area-context', () => mockSafeAreaContext);
jest.mock('react-native-device-info', () => require('react-native-device-info/jest/react-native-device-info-mock'));
jest.mock('@react-native-documents/picker', () => ({
  pick: jest.fn(async () => [{uri: 'content://picked/voice.onnx', name: 'voice.onnx', error: null}]),
  keepLocalCopy: jest.fn(async ({files}: {files: Array<{fileName: string}>}) => {
    const fs = require('./mocks/fs');
    const path = `${fs.DocumentDirectoryPath}/${files[0].fileName}`;
    fs.__putFile(path, 1234);
    return [{status: 'success', sourceUri: 'content://picked', localUri: `file://${path}`}];
  }),
  types: {allFiles: '*/*', audio: 'audio/*'},
  errorCodes: {OPERATION_CANCELED: 'OPERATION_CANCELED'},
  isErrorWithCode: (e: any) => !!e?.code,
}));
require('llama.rn/jest/mock');
