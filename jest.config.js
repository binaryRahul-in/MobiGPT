module.exports = {
  preset: 'react-native',
  setupFiles: ['./jest/setup.ts'],
  testPathIgnorePatterns: ['/node_modules/', '/android/', '/ios/', '/packages/.*/(android|ios|cpp)/'],
  transformIgnorePatterns: [
    'node_modules/(?!((jest-)?react-native|@react-native(-community)?|@react-navigation|react-native-paper|react-native-vector-icons|react-native-safe-area-context|react-native-screens|react-native-svg|@dr.pogodin|@react-native-documents|llama.rn|mobx-persist-store|mobx-react-lite)/)',
  ],
  moduleNameMapper: {
    // Use the React Native entry (no react-dom batching) like Metro does.
    '^mobx-react-lite$': '<rootDir>/node_modules/mobx-react-lite/es/index.js',
    '^react-native-mobigpt-voice$': '<rootDir>/jest/mocks/voice.ts',
    '^react-native-mobigpt-device$': '<rootDir>/jest/mocks/device.ts',
    '^@dr.pogodin/react-native-fs$': '<rootDir>/jest/mocks/fs.ts',
    '^@react-native-community/slider$': '<rootDir>/jest/mocks/slider.tsx',
    '^react-native-svg$': '<rootDir>/jest/mocks/svg.tsx',
  },
  collectCoverageFrom: ['src/**/*.{ts,tsx}', '!src/**/*.d.ts'],
  coverageReporters: ['text-summary', 'lcov'],
  testTimeout: 20000,
};
