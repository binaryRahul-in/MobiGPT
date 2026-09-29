import {makeAutoObservable, runInAction} from 'mobx';
import DeviceInfo from 'react-native-device-info';

import {checkForAppUpdate, UpdateCheck} from '../services/updates';

export class UpdateStore {
  result: UpdateCheck | null = null;
  checking = false;
  error: string | null = null;
  checkedAt = 0;

  constructor(private currentVersion: () => string = () => DeviceInfo.getVersion()) {
    makeAutoObservable(this, {}, {autoBind: true});
  }

  get version(): string {
    return this.currentVersion();
  }

  async check(): Promise<UpdateCheck | null> {
    this.checking = true;
    this.error = null;
    try {
      const r = await checkForAppUpdate(this.currentVersion());
      runInAction(() => {
        this.result = r;
        this.checkedAt = Date.now();
      });
      return r;
    } catch (e: any) {
      runInAction(() => (this.error = e?.message ?? String(e)));
      return null;
    } finally {
      runInAction(() => (this.checking = false));
    }
  }
}
