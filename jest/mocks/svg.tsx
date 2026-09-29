import React from 'react';
import {View} from 'react-native';

const make = (name: string) => {
  const C = ({children}: {children?: React.ReactNode}) => <View testID={`svg-${name}`}>{children}</View>;
  C.displayName = name;
  return C;
};

const Svg = make('Svg');
export default Svg;
export const Circle = make('Circle');
export const Defs = make('Defs');
export const LinearGradient = make('LinearGradient');
export const Path = make('Path');
export const Rect = make('Rect');
export const Stop = make('Stop');
