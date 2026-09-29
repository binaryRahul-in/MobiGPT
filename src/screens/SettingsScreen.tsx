import Slider from '@react-native-community/slider';
import {observer} from 'mobx-react-lite';
import React, {useState} from 'react';
import {Alert, StyleSheet, View} from 'react-native';
import {Button, List, SegmentedButtons, Switch, Text, TextInput, useTheme} from 'react-native-paper';

import {Screen, Section, StatusChip} from '../components/ui';
import {RootScreenProps} from '../navigation/types';
import {useStores} from '../stores/RootStore';
import {LlmSettings} from '../stores/SettingsStore';
import {spacing} from '../theme';

export const SettingsScreen = observer(function SettingsScreen({navigation}: RootScreenProps<'Settings'>) {
  const theme = useTheme();
  const {settings, device, models, chat} = useStores();
  const s = settings.llm;
  const [token, setToken] = useState(settings.hfToken);
  const [prompt, setPrompt] = useState(s.systemPrompt);
  const changedWhileLoaded = (patch: Partial<LlmSettings>) => {
    settings.updateLlm(patch);
  };

  return (
    <Screen testID="settings-screen">
      <Section title="Appearance" icon="palette-outline">
        <SegmentedButtons
          value={settings.themeMode}
          onValueChange={v => settings.setTheme(v as typeof settings.themeMode)}
          buttons={[
            {value: 'system', label: 'System', icon: 'theme-light-dark'},
            {value: 'light', label: 'Light', icon: 'white-balance-sunny'},
            {value: 'dark', label: 'Dark', icon: 'weather-night'},
          ]}
        />
      </Section>

      <Section title="Inference hardware" icon="chip" subtitle="Applied next time a model is loaded">
        <SegmentedButtons
          value={s.accel}
          onValueChange={v => changedWhileLoaded({accel: v as LlmSettings['accel']})}
          buttons={[
            {value: 'cpu', label: 'CPU', icon: 'cpu-64-bit'},
            {value: 'gpu', label: 'GPU', icon: 'expansion-card', disabled: !device.gpu.supported},
            {value: 'npu', label: 'NPU', icon: 'brain', disabled: !device.npu.supported},
          ]}
        />
        {s.accel === 'gpu' && !device.gpu.supported ? <StatusChip severity="block" label={device.gpu.reason} /> : null}
        {s.accel === 'npu' ? <StatusChip severity="warn" label="Experimental: unsupported ops fall back to CPU" /> : null}
        {s.accel !== 'cpu' ? (
          <SliderRow
            label="GPU/NPU layers"
            value={s.gpuLayers}
            min={0}
            max={99}
            step={1}
            onChange={v => changedWhileLoaded({gpuLayers: v})}
          />
        ) : null}
        <SliderRow
          label="CPU threads (0 = auto)"
          value={s.nThreads}
          min={0}
          max={Math.max(4, device.profile.cores)}
          step={1}
          onChange={v => changedWhileLoaded({nThreads: v})}
        />
        <List.Item
          title="Flash attention"
          description="Faster + smaller KV cache on supported backends"
          right={() => <Switch value={s.flashAttn} onValueChange={v => changedWhileLoaded({flashAttn: v})} />}
        />
        <Text variant="labelMedium">KV cache type</Text>
        <SegmentedButtons
          value={s.cacheType}
          onValueChange={v => changedWhileLoaded({cacheType: v as LlmSettings['cacheType']})}
          buttons={[
            {value: 'f16', label: 'F16'},
            {value: 'q8_0', label: 'Q8_0', disabled: !s.flashAttn},
            {value: 'q4_0', label: 'Q4_0', disabled: !s.flashAttn},
          ]}
        />
        <List.Item
          title="Lock model in RAM (mlock)"
          description="Prevents swapping; may fail on low-RAM phones"
          right={() => <Switch value={s.useMlock} onValueChange={v => changedWhileLoaded({useMlock: v})} />}
        />
      </Section>

      <Section title="Generation" icon="tune" subtitle="Used for every chat message">
        <SliderRow label="Context length" value={s.nCtx} min={512} max={16384} step={512} onChange={v => changedWhileLoaded({nCtx: v})} />
        <SliderRow
          label="Max new tokens"
          value={s.maxTokens}
          min={64}
          max={4096}
          step={64}
          onChange={v => settings.updateLlm({maxTokens: v})}
        />
        <SliderRow
          label="Temperature"
          value={s.temperature}
          min={0}
          max={2}
          step={0.05}
          decimals={2}
          onChange={v => settings.updateLlm({temperature: v})}
        />
        <SliderRow label="Top-p" value={s.topP} min={0.1} max={1} step={0.01} decimals={2} onChange={v => settings.updateLlm({topP: v})} />
        <SliderRow label="Min-p" value={s.minP} min={0} max={0.5} step={0.01} decimals={2} onChange={v => settings.updateLlm({minP: v})} />
        <SliderRow
          label="Repeat penalty"
          value={s.repeatPenalty}
          min={1}
          max={1.5}
          step={0.01}
          decimals={2}
          onChange={v => settings.updateLlm({repeatPenalty: v})}
        />
        <List.Item
          title="Thinking mode"
          description="For hybrid reasoning models (Qwen3…)"
          right={() => <Switch value={s.enableThinking} onValueChange={v => settings.updateLlm({enableThinking: v})} />}
        />
        <TextInput
          mode="outlined"
          label="System prompt"
          multiline
          value={prompt}
          onChangeText={setPrompt}
          onBlur={() => settings.updateLlm({systemPrompt: prompt})}
        />
        <Button onPress={() => settings.resetLlm()}>Reset to defaults</Button>
        {models.loaded ? (
          <Button mode="contained-tonal" icon="reload" onPress={() => models.load(models.loaded!.id, {force: true}).catch(() => undefined)}>
            Reload model with new settings
          </Button>
        ) : null}
      </Section>

      <Section title="Hugging Face" icon="key-outline" subtitle="Optional token for gated models (read access is enough)">
        <TextInput
          mode="outlined"
          label="Access token"
          secureTextEntry
          value={token}
          onChangeText={setToken}
          onBlur={() => settings.setHfToken(token)}
          autoCapitalize="none"
          autoCorrect={false}
        />
      </Section>

      <Section title="Behaviour" icon="cog-outline">
        <List.Item
          title="Load last model on start"
          right={() => <Switch value={settings.autoLoadLastModel} onValueChange={settings.setAutoLoad} />}
        />
        <List.Item
          title="Check for app updates on start"
          right={() => <Switch value={settings.checkUpdatesOnLaunch} onValueChange={settings.setCheckUpdates} />}
        />
        <List.Item
          title="Features"
          description="Install / remove optional modules"
          left={p => <List.Icon {...p} icon="puzzle-outline" />}
          onPress={() => navigation.navigate('Features')}
        />
        <List.Item
          title="About & updates"
          left={p => <List.Icon {...p} icon="information-outline" />}
          onPress={() => navigation.navigate('About')}
          testID="open-about"
        />
      </Section>

      <Section title="Data" icon="database-outline">
        <Button
          textColor={theme.colors.error}
          onPress={() =>
            Alert.alert('Delete all conversations?', 'This cannot be undone.', [
              {text: 'Cancel'},
              {text: 'Delete', style: 'destructive', onPress: chat.clearAll},
            ])
          }
        >
          Delete all conversations
        </Button>
        <Button onPress={settings.resetOnboarding}>Replay introduction</Button>
      </Section>
    </Screen>
  );
});

function SliderRow({
  label,
  value,
  min,
  max,
  step,
  decimals = 0,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  decimals?: number;
  onChange: (v: number) => void;
}) {
  const theme = useTheme();
  const [local, setLocal] = useState(value);
  return (
    <View style={styles.slider}>
      <View style={styles.sliderHead}>
        <Text variant="bodyMedium">{label}</Text>
        <Text variant="bodyMedium" style={styles.bold}>
          {local.toFixed(decimals)}
        </Text>
      </View>
      <Slider
        minimumValue={min}
        maximumValue={max}
        step={step}
        value={value}
        onValueChange={setLocal}
        onSlidingComplete={onChange}
        minimumTrackTintColor={theme.colors.primary}
        thumbTintColor={theme.colors.primary}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  slider: {gap: 2, marginTop: spacing.xs},
  sliderHead: {flexDirection: 'row', justifyContent: 'space-between'},
  bold: {fontWeight: '700'},
});
