import NativeMobiGPTVoice from './NativeMobiGPTVoice';
import type {VoiceJsi} from './types';

export * from './types';
export type {TtsVoice, DecodedAudio} from './NativeMobiGPTVoice';

const g = globalThis as typeof globalThis & {__MobiGPTVoice?: VoiceJsi};

let installPromise: Promise<VoiceJsi | null> | null = null;

/**
 * True when the native module was compiled into this build. The voice
 * feature can be excluded at build time (see mobigpt.features.json), in which
 * case the whole feature is hidden instead of crashing.
 */
export function isVoiceModuleAvailable(): boolean {
  return NativeMobiGPTVoice != null;
}

/** Installs the JSI bindings once and returns them (null when unavailable). */
export function getVoiceEngine(): Promise<VoiceJsi | null> {
  if (g.__MobiGPTVoice) {
    return Promise.resolve(g.__MobiGPTVoice);
  }
  if (!installPromise) {
    installPromise = (async () => {
      if (!NativeMobiGPTVoice) {
        return null;
      }
      const ok = await NativeMobiGPTVoice.install();
      // The install is scheduled on the JS thread; give it one tick.
      for (let i = 0; i < 20 && !g.__MobiGPTVoice; i++) {
        await new Promise<void>(r => setTimeout(r, 25));
      }
      if (!ok || !g.__MobiGPTVoice) {
        installPromise = null;
        return null;
      }
      return g.__MobiGPTVoice;
    })();
  }
  return installPromise;
}

export const VoiceNative = NativeMobiGPTVoice;
