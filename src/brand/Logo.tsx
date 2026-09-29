import React from 'react';
import {Text} from 'react-native-paper';
import Svg, {Circle, Defs, LinearGradient, Path, Rect, Stop} from 'react-native-svg';

import {brand} from '../theme';

/**
 * MobiGPT mark: an "M" drawn as a voice waveform inside a chip-shaped tile —
 * chat + voice, on your device.
 */
export function Logo({size = 64, testID}: {size?: number; testID?: string}) {
  return (
    <Svg width={size} height={size} viewBox="0 0 100 100" testID={testID}>
      <Defs>
        <LinearGradient id="mg" x1="0" y1="0" x2="1" y2="1">
          <Stop offset="0" stopColor={brand.gradient[0]} />
          <Stop offset="1" stopColor={brand.gradient[1]} />
        </LinearGradient>
      </Defs>
      <Rect x="4" y="4" width="92" height="92" rx="26" fill="url(#mg)" />
      <Path
        d="M24 68 L24 38 Q24 32 29 36 L46 55 Q50 59 54 55 L71 36 Q76 32 76 38 L76 68"
        stroke="#FFFFFF"
        strokeWidth="9"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
      <Circle cx="78" cy="22" r="6" fill="#FFFFFF" opacity={0.9} />
    </Svg>
  );
}

/** Kept as text (not SVG) so it respects accessibility font scaling. */
export function Wordmark({color}: {color: string}) {
  return (
    <Text variant="headlineMedium" style={{color, fontWeight: '800', letterSpacing: 0.5}}>
      Mobi
      <Text variant="headlineMedium" style={{color: brand.teal, fontWeight: '800'}}>
        GPT
      </Text>
    </Text>
  );
}
