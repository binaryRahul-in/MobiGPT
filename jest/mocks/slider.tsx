import React from 'react';
import {View} from 'react-native';

export default function Slider(props: {testID?: string}) {
  return <View testID={props.testID ?? 'slider'} />;
}
