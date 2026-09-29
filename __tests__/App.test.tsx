import {act, fireEvent, render, screen, waitFor} from '@testing-library/react-native';
import React from 'react';

import * as fs from '../jest/mocks/fs';
import {profile} from '../jest/fixtures';
import App from '../src/App';
import {RootStore} from '../src/stores/RootStore';

jest.useRealTimers();

function renderApp(over = {}, onboarded = false) {
  const store = new RootStore({persist: false, probe: async () => profile(over)});
  if (onboarded) {
    store.settings.completeOnboarding();
  }
  const utils = render(<App store={store} />);
  return {store, ...utils};
}

beforeEach(() => {
  fs.__reset();
  (global as any).fetch = jest.fn(async () => ({ok: false, status: 404, json: async () => ({})}));
});

describe('MobiGPT app', () => {
  it('walks through the branded onboarding and lands on the tabs', async () => {
    const {store} = renderApp();
    expect(await screen.findByTestId('onboarding-welcome')).toBeTruthy();
    expect(screen.getByText('Your private AI, running entirely on this phone.')).toBeTruthy();
    await waitFor(() => expect(store.device.probed).toBe(true));

    // Feature toggles reflect hardware checks.
    fireEvent.press(screen.getByTestId('onboarding-next'));
    fireEvent.press(screen.getByTestId('onboarding-next'));
    fireEvent.press(screen.getByTestId('onboarding-next'));
    const voiceToggle = await screen.findByTestId('onboarding-toggle-voice');
    fireEvent(voiceToggle, 'valueChange', true);
    expect(store.settings.isEnabled('voice')).toBe(true);

    fireEvent.press(screen.getByTestId('onboarding-finish'));
    expect(await screen.findByTestId('chat-empty')).toBeTruthy();
    expect(store.settings.onboardingDone).toBe(true);
  });

  it('shows device capabilities and model compatibility', async () => {
    const {store} = renderApp({totalRam: 6e9}, true);
    await waitFor(() => expect(store.device.probed).toBe(true));
    fireEvent.press(await screen.findByTestId('tab-device'));
    expect(await screen.findByTestId('device-model')).toHaveTextContent('Pixel');
    expect(screen.getByTestId('device-ram')).toHaveTextContent(/6\.0 GB/);
    expect(screen.getByText('Mid-range (6 GB) · runs Q4 models up to ~3.2B parameters')).toBeTruthy();

    fireEvent.press(screen.getByTestId('tab-models'));
    expect(await screen.findByTestId('model-qwen3-0.6b')).toBeTruthy();
    expect(screen.getByTestId('download-qwen3-0.6b')).toBeTruthy();
  });

  it('guides the user to enable Voice Studio with hardware warnings', async () => {
    const {store} = renderApp({totalRam: 4e9}, true);
    await waitFor(() => expect(store.device.probed).toBe(true));
    fireEvent.press(await screen.findByTestId('tab-voice'));
    expect(await screen.findByTestId('voice-disabled')).toBeTruthy();
    expect(screen.getByText(/6 GB RAM recommended/)).toBeTruthy();

    fireEvent.press(screen.getByTestId('voice-enable'));
    const toggle = await screen.findByTestId('feature-toggle-voice');
    fireEvent(toggle, 'valueChange', true);
    // 4 GB triggers a confirmation dialog listing the caveats.
    fireEvent.press(await screen.findByTestId('feature-confirm'));
    await waitFor(() => expect(store.settings.isEnabled('voice')).toBe(true));
    expect(await screen.findByTestId('voice-screen')).toBeTruthy();
    expect(screen.getByTestId('voice-missing')).toBeTruthy();
  });

  it('exposes locked mobile optimisations in the voice options', async () => {
    const {store} = renderApp({totalRam: 8e9}, true);
    await waitFor(() => expect(store.device.probed).toBe(true));
    act(() => store.settings.setFeature('voice', true));
    fireEvent.press(await screen.findByTestId('open-settings'));
    fireEvent.press(await screen.findByText('Features'));
    expect(await screen.findByText('Retrieval index (FAISS)')).toBeTruthy();
    expect(screen.getByText(/index_rate is hard-wired to 0/)).toBeTruthy();
    expect(screen.getByText('Native tensor path')).toBeTruthy();
    fireEvent.press(screen.getByTestId('option-pitchMethod-fcpe'));
    expect(store.settings.voice?.pitchMethod).toBe('fcpe');
  });

  it('opens About with the version and update check', async () => {
    const {store} = renderApp({}, true);
    await waitFor(() => expect(store.device.probed).toBe(true));
    fireEvent.press(await screen.findByTestId('open-settings'));
    fireEvent.press(await screen.findByTestId('open-about'));
    expect(await screen.findByTestId('about-version')).toBeTruthy();
    expect(screen.getByText(/PocketPal AI/)).toBeTruthy();
  });
});
