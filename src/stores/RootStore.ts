import {createContext, useContext} from 'react';
import {reaction} from 'mobx';

import {DeviceProfile} from '../features/device';
import {recommendedVoiceSettings} from '../features/registry';
import {probeDevice} from '../services/deviceProbe';
import {ensureDirs} from '../services/paths';
import {BenchmarkStore} from './BenchmarkStore';
import {ChatStore} from './ChatStore';
import {DeviceStore} from './DeviceStore';
import {ModelStore} from './ModelStore';
import {SettingsStore} from './SettingsStore';
import {TtsStore} from './TtsStore';
import {UpdateStore} from './UpdateStore';
import {VoiceStore} from './VoiceStore';

export class RootStore {
  settings: SettingsStore;
  device: DeviceStore;
  models: ModelStore;
  chat: ChatStore;
  voice: VoiceStore;
  tts: TtsStore;
  bench: BenchmarkStore;
  updates: UpdateStore;

  constructor(opts: {persist?: boolean; probe?: () => Promise<DeviceProfile>} = {}) {
    const persist = opts.persist ?? true;
    this.settings = new SettingsStore(persist);
    this.device = new DeviceStore(opts.probe ?? probeDevice);
    this.models = new ModelStore(this.settings, () => this.device.profile, persist);
    this.chat = new ChatStore(this.settings, this.models, persist);
    this.voice = new VoiceStore(this.settings, undefined, persist);
    this.tts = new TtsStore(this.settings, undefined, persist);
    this.bench = new BenchmarkStore(this.settings, this.device, this.models, this.voice, persist);
    this.updates = new UpdateStore();

    // Keep HF auth in sync for every client.
    reaction(
      () => this.settings.hfToken,
      t => {
        this.models.hf.setToken(t);
      },
    );
  }

  /** App start-up: probe hardware, apply device-aware defaults, warm caches. */
  async bootstrap() {
    await ensureDirs().catch(() => undefined);
    const profile = await this.device.refresh();
    if (!this.settings.voice) {
      this.settings.setVoice(recommendedVoiceSettings(profile));
    }
    this.models.refreshCatalog().catch(() => undefined);
    if (this.settings.checkUpdatesOnLaunch) {
      this.updates.check().catch(() => undefined);
    }
    this.models.autoLoad().catch(() => undefined);
  }
}

export const StoreContext = createContext<RootStore | null>(null);

export function useStores(): RootStore {
  const s = useContext(StoreContext);
  if (!s) {
    throw new Error('useStores must be used inside <StoreContext.Provider>');
  }
  return s;
}
