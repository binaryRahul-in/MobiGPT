import Slider from '@react-native-community/slider';
import {observer} from 'mobx-react-lite';
import React, {useState} from 'react';
import {StyleSheet, View} from 'react-native';
import {ActivityIndicator, Button, Chip, SegmentedButtons, Switch, Text, TextInput, useTheme} from 'react-native-paper';

import {DownloadProgress, Section} from '../../components/ui';
import {useStores} from '../../stores/RootStore';
import {ttsCatalog} from '../../stores/TtsStore';
import {spacing} from '../../theme';
import {formatBytes} from '../../utils/format';

/** Voice Studio "Text" mode: neural (Kokoro) or system TTS, optionally converted into the selected RVC voice. */
export const TtsPanel = observer(function TtsPanel({onMessage}: {onMessage: (m: string) => void}) {
  const theme = useTheme();
  const {tts, voice} = useStores();
  const [text, setText] = useState('Hello! This voice was generated entirely on my phone.');
  const neural = tts.engine === 'neural';
  const convert = tts.convertWithRvc && voice.ready;
  const working = tts.busy || voice.converting;

  const speak = async () => {
    try {
      let path: string;
      if (neural) {
        path = (await tts.synthesize(text)).path;
        if (convert) {
          path = (await voice.convert(path, `“${text.slice(0, 40)}${text.length > 40 ? '…' : ''}”`)).outputPath;
        }
      } else if (convert) {
        path = (await voice.convertText(text)).outputPath;
      } else {
        path = await voice.synthesizeSystem(text);
      }
      voice.play(path).catch(e => onMessage(e?.message ?? String(e)));
    } catch (e: any) {
      onMessage(e?.message ?? String(e));
    }
  };

  return (
    <Section
      title="Text → speech"
      subtitle={neural ? 'Kokoro-82M neural voice, on-device' : 'System text-to-speech (offline voices)'}
      icon="text-to-speech"
      testID="tts-panel"
    >
      <SegmentedButtons
        value={tts.engine}
        onValueChange={v => tts.setEngine(v as 'neural' | 'system')}
        buttons={[
          {value: 'neural', label: 'Neural', icon: 'brain', testID: 'tts-engine-neural'},
          {value: 'system', label: 'System', icon: 'cellphone-sound', testID: 'tts-engine-system'},
        ]}
      />

      {neural ? (
        <>
          <View style={styles.chips}>
            {ttsCatalog.voices.map(v => (
              <Chip
                key={v.id}
                compact
                selected={v.id === tts.selectedVoiceId}
                showSelectedCheck
                icon={v.gender === 'female' ? 'face-woman-outline' : 'face-man-outline'}
                onPress={() => tts.setVoice(v.id)}
                testID={`tts-voice-${v.id}`}
              >
                {v.name} · {v.accent.startsWith('British') ? 'UK' : 'US'}
              </Chip>
            ))}
          </View>
          {!tts.ready ? <TtsInstall /> : null}
          <View style={styles.row}>
            <Text variant="bodySmall">Speed {tts.speed.toFixed(2)}×</Text>
            <Slider
              style={styles.flex}
              minimumValue={0.5}
              maximumValue={2}
              step={0.05}
              value={tts.speed}
              onSlidingComplete={tts.setSpeed}
              minimumTrackTintColor={theme.colors.primary}
            />
          </View>
        </>
      ) : null}

      <TextInput mode="outlined" multiline value={text} onChangeText={setText} style={styles.input} testID="tts-text" />

      <View style={styles.row}>
        <Switch value={convert} disabled={!voice.ready} onValueChange={tts.setConvertWithRvc} testID="tts-convert-switch" />
        <Text variant="bodyMedium" style={styles.flex}>
          {voice.ready
            ? `Convert into ${voice.selectedVoice?.name ?? 'the selected voice'} (RVC)`
            : 'Install the Voice Studio packs to also convert into an RVC voice'}
        </Text>
      </View>

      <Button
        mode="contained"
        icon={working ? undefined : 'play'}
        loading={working}
        disabled={working || !text.trim() || (neural && !tts.ready)}
        onPress={speak}
        testID="tts-speak"
      >
        {working ? (tts.busy ? 'Synthesising…' : 'Converting…') : convert ? 'Speak in this voice' : 'Speak'}
      </Button>

      {neural && tts.lastResult ? (
        <Text variant="bodySmall" style={{color: theme.colors.onSurfaceVariant}} testID="tts-result">
          {tts.lastResult.seconds.toFixed(1)} s of speech in {(tts.lastResult.inferMs / 1000).toFixed(1)} s · RTF{' '}
          {tts.lastResult.realtimeFactor.toFixed(2)}×
          {tts.lastUnknown.length ? ` · spelled out: ${tts.lastUnknown.slice(0, 5).join(', ')}` : ''}
        </Text>
      ) : null}
      {tts.error ? (
        <Text variant="bodySmall" style={{color: theme.colors.error}}>
          {tts.error}
        </Text>
      ) : null}
    </Section>
  );
});

const TtsInstall = observer(function TtsInstall() {
  const theme = useTheme();
  const {tts} = useStores();
  const step = tts.installing;
  const task = step ? tts.downloads.get(step.id) : undefined;
  if (step) {
    return (
      <View style={styles.box} testID="tts-install-progress">
        <Text variant="labelLarge">
          Step {step.index} of {step.total} · {step.label}
        </Text>
        {task && task.state === 'downloading' ? (
          <DownloadProgress bytes={task.bytes} total={task.total} speedBps={task.speedBps} onCancel={() => tts.downloads.cancel(step.id)} />
        ) : (
          <View style={styles.row}>
            <ActivityIndicator size="small" />
            <Text variant="bodySmall">Connecting…</Text>
          </View>
        )}
      </View>
    );
  }
  return (
    <View style={styles.box}>
      <Text variant="bodySmall" style={{color: theme.colors.onSurfaceVariant}}>
        Needs {tts.plan.map(p => p.label.split(' · ')[0].toLowerCase()).join(', ')} · {formatBytes(tts.missingBytes)} ·{' '}
        {ttsCatalog.models[0].license}
      </Text>
      {tts.installError ? (
        <Text variant="bodySmall" style={{color: theme.colors.error}} testID="tts-install-error">
          {tts.installError}
        </Text>
      ) : null}
      <Button
        mode="contained-tonal"
        icon={tts.installError ? 'refresh' : 'download'}
        onPress={() => tts.install().catch(() => undefined)}
        testID="tts-install"
      >
        {tts.installError ? 'Retry' : `Install neural voice (${formatBytes(tts.missingBytes)})`}
      </Button>
    </View>
  );
});

const styles = StyleSheet.create({
  flex: {flex: 1},
  chips: {flexDirection: 'row', flexWrap: 'wrap', gap: 6},
  row: {flexDirection: 'row', alignItems: 'center', gap: spacing.sm},
  box: {gap: spacing.sm},
  input: {minHeight: 96},
});
