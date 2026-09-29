// Autolinking config. The optional voice module (ONNX Runtime + RVC engine,
// ~20 MB per ABI) can be compiled out entirely by setting
// `voice.enabled: false` in mobigpt.features.json; the app then hides the
// Voice Studio instead of crashing (see src/features/registry.ts).
const features = require('./mobigpt.features.json');

const voiceEnabled = process.env.MOBIGPT_VOICE ? process.env.MOBIGPT_VOICE !== '0' : features.voice?.enabled !== false;

module.exports = {
  project: {
    ios: {},
    android: {},
  },
  assets: [],
  dependencies: voiceEnabled
    ? {}
    : {
        'react-native-mobigpt-voice': {
          platforms: {android: null, ios: null},
        },
      },
};
