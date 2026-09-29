import type {TurboModule} from 'react-native';
import {TurboModuleRegistry} from 'react-native';

export type TtsVoice = {
  id: string;
  name: string;
  language: string;
  quality: number;
  requiresNetwork: boolean;
};

export type DecodedAudio = {
  path: string;
  sampleRate: number;
  seconds: number;
};

export interface Spec extends TurboModule {
  /** Loads the native library and installs `global.__MobiGPTVoice`. */
  install(): Promise<boolean>;
  /** Renders text with the OS speech engine into a WAV file (offline voices when available). */
  synthesizeSpeech(text: string, outputPath: string, language: string, voiceId: string, rate: number, pitch: number): Promise<DecodedAudio>;
  listTtsVoices(): Promise<TtsVoice[]>;
  /** Decodes any OS-supported audio file (mp3, m4a, ogg, flac, wav…) into 16-bit mono WAV. */
  decodeAudioToWav(inputPath: string, outputPath: string): Promise<DecodedAudio>;
}

export default TurboModuleRegistry.get<Spec>('MobiGPTVoice');
