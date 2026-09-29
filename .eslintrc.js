module.exports = {
  root: true,
  extends: '@react-native',
  rules: {
    'react-native/no-inline-styles': 'off',
    'react/no-unstable-nested-components': ['warn', {allowAsProps: true}],
  },
  overrides: [
    {
      // `const X = observer(function X() {...})` keeps component names in
      // React DevTools; the inner name intentionally shadows the const.
      files: ['*.tsx'],
      rules: {'@typescript-eslint/no-shadow': 'off'},
    },
    {
      files: ['jest/**', '__tests__/**', '**/*.test.ts', '**/*.test.tsx'],
      env: {jest: true},
    },
  ],
};
