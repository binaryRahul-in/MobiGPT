import {DarkTheme as NavDark, DefaultTheme as NavLight, Theme as NavTheme} from '@react-navigation/native';
import {MD3DarkTheme, MD3LightTheme, MD3Theme} from 'react-native-paper';

/** MobiGPT brand palette. */
export const brand = {
  violet: '#6D5BFF',
  violetDark: '#8F82FF',
  teal: '#00C9A7',
  coral: '#FF8A5B',
  ink: '#0B0C14',
  night: '#151726',
  dusk: '#1E2135',
  mist: '#F5F5FB',
  gradient: ['#6D5BFF', '#00C9A7'] as const,
};

export const severityColors = {
  ok: '#1DB980',
  info: '#3B82F6',
  warn: '#F5A524',
  block: '#EF4444',
};

export const lightTheme: MD3Theme = {
  ...MD3LightTheme,
  roundness: 4,
  colors: {
    ...MD3LightTheme.colors,
    primary: brand.violet,
    onPrimary: '#FFFFFF',
    primaryContainer: '#E6E2FF',
    onPrimaryContainer: '#1E1466',
    secondary: '#00A88C',
    onSecondary: '#FFFFFF',
    secondaryContainer: '#C8F5EB',
    onSecondaryContainer: '#00382E',
    tertiary: brand.coral,
    tertiaryContainer: '#FFE3D8',
    background: brand.mist,
    surface: '#FFFFFF',
    surfaceVariant: '#ECEBF7',
    onSurfaceVariant: '#4A4860',
    outline: '#C9C7DA',
    elevation: {
      ...MD3LightTheme.colors.elevation,
      level1: '#FFFFFF',
      level2: '#F9F8FF',
      level3: '#F3F1FF',
    },
  },
};

export const darkTheme: MD3Theme = {
  ...MD3DarkTheme,
  roundness: 4,
  colors: {
    ...MD3DarkTheme.colors,
    primary: brand.violetDark,
    onPrimary: '#140C52',
    primaryContainer: '#3A2FA8',
    onPrimaryContainer: '#E6E2FF',
    secondary: brand.teal,
    onSecondary: '#00382E',
    secondaryContainer: '#00524A',
    onSecondaryContainer: '#C8F5EB',
    tertiary: brand.coral,
    tertiaryContainer: '#5C2A17',
    background: brand.ink,
    surface: brand.night,
    surfaceVariant: brand.dusk,
    onSurfaceVariant: '#B7B5CC',
    outline: '#3A3D55',
    elevation: {
      ...MD3DarkTheme.colors.elevation,
      level1: brand.night,
      level2: '#191B2C',
      level3: brand.dusk,
    },
  },
};

export function navigationTheme(paper: MD3Theme, dark: boolean): NavTheme {
  const base = dark ? NavDark : NavLight;
  return {
    ...base,
    colors: {
      ...base.colors,
      primary: paper.colors.primary,
      background: paper.colors.background,
      card: paper.colors.surface,
      text: paper.colors.onSurface,
      border: paper.colors.outline,
      notification: paper.colors.tertiary,
    },
  };
}

export const spacing = {xs: 4, sm: 8, md: 12, lg: 16, xl: 24, xxl: 32};
