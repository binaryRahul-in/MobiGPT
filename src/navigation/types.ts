import type {BottomTabScreenProps} from '@react-navigation/bottom-tabs';
import type {CompositeScreenProps, NavigatorScreenParams} from '@react-navigation/native';
import type {NativeStackScreenProps} from '@react-navigation/native-stack';

export type TabParamList = {
  Chat: undefined;
  Models: undefined;
  Voice: undefined;
  Device: undefined;
};

export type RootStackParamList = {
  Onboarding: undefined;
  Main: NavigatorScreenParams<TabParamList>;
  HFBrowser: undefined;
  VoiceLibrary: {tab?: 'installed' | 'presets' | 'hub' | 'packs'} | undefined;
  Benchmark: undefined;
  Features: {focus?: string} | undefined;
  Settings: undefined;
  About: undefined;
  History: undefined;
};

export type RootScreenProps<T extends keyof RootStackParamList> = NativeStackScreenProps<RootStackParamList, T>;
export type TabScreenProps<T extends keyof TabParamList> = CompositeScreenProps<
  BottomTabScreenProps<TabParamList, T>,
  NativeStackScreenProps<RootStackParamList>
>;
