import Slider from '@react-native-community/slider';
import {observer} from 'mobx-react-lite';
import React, {useState} from 'react';
import {Alert, ScrollView, StyleSheet, View} from 'react-native';
import {
  ActivityIndicator,
  Button,
  Chip,
  IconButton,
  List,
  ProgressBar,
  SegmentedButtons,
  Snackbar,
  Text,
  useTheme,
} from 'react-native-paper';

import {DownloadProgress, EmptyState, IssueList, LevelMeter, Screen, Section, StatTile, StatusChip, TileRow} from '../components/ui';
import {TabScreenProps} from '../navigation/types';
import {pickAndImport} from '../services/importFile';
import {ensureDirs, Paths} from '../services/paths';
import {useStores} from '../stores/RootStore';
import {voicesCatalog} from '../stores/VoiceStore';
import {spacing} from '../theme';
import {formatBytes, formatDuration} from '../utils/format';
import {TtsPanel} from './voice/TtsPanel';

type Mode = 'record' | 'file' | 'text' | 'live';

export const VoiceScreen = observer(function VoiceScreen({navigation}: TabScreenProps<'Voice'>) {
  const theme = useTheme();
  const {voice, settings, device} = useStores();
  const [mode, setMode] = useState<Mode>('record');
  const [msg, setMsg] = useState<string | null>(null);

  const guard = async (fn: () => Promise<unknown>, done?: string) => {
    try {
      await fn();
      if (done) {
        setMsg(done);
      }
    } catch (e: any) {
      setMsg(e?.message ?? String(e));
    }
  };

  if (!device.profile.voiceModuleAvailable && device.probed) {
    return (
      <Screen>
        <EmptyState
          icon="puzzle-remove-outline"
          title="Voice engine not in this build"
          body="This build was compiled with voice.enabled = false in mobigpt.features.json. Rebuild with it enabled to use Voice Studio."
        />
      </Screen>
    );
  }

  if (!settings.isEnabled('voice')) {
    const ev = device.evaluation('voice');
    return (
      <Screen testID="voice-disabled">
        <EmptyState
          icon="account-voice"
          title="Voice Studio"
          body="Convert your speech — or any text — into another voice with RVC, entirely on-device. Enable the feature to download the engine packs (~200 MB with INT8)."
          action={
            <View style={styles.center}>
              <StatusChip severity={ev.status} />
              <IssueList issues={ev.issues} />
              <Button
                mode="contained"
                icon="puzzle-plus-outline"
                onPress={() => navigation.navigate('Features', {focus: 'voice'})}
                testID="voice-enable"
              >
                Set up Voice Studio
              </Button>
            </View>
          }
        />
      </Screen>
    );
  }

  const v = settings.voice;
  const liveEval = device.evaluation('liveVoice');

  return (
    <View style={styles.flex} testID="voice-screen">
      <Screen>
        <Section
          title="Voice"
          subtitle={
            voice.selectedVoice
              ? `${voice.selectedVoice.version} · ${voice.selectedVoice.sampleRate / 1000} kHz · ${
                  voice.selectedVoice.usesF0 ? 'pitch-guided' : 'no-f0'
                }`
              : 'No voice installed yet'
          }
          icon="account-music"
          right={
            <Button onPress={() => navigation.navigate('VoiceLibrary')} testID="voice-library">
              Library
            </Button>
          }
        >
          <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.chips}>
            {voice.voices.map(x => (
              <Chip key={x.id} selected={x.id === voice.selectedVoice?.id} showSelectedCheck onPress={() => voice.selectVoice(x.id)}>
                {x.name}
              </Chip>
            ))}
            <Chip icon="plus" onPress={() => navigation.navigate('VoiceLibrary', {tab: 'presets'})}>
              Add voice
            </Chip>
          </ScrollView>
        </Section>

        {!voice.ready ? (
          <Section
            title="Almost ready"
            subtitle="Install what's missing for your current settings"
            icon="package-down"
            testID="voice-missing"
          >
            {voice.missing.map(m => (
              <Text key={m} variant="bodyMedium">
                • {m}
              </Text>
            ))}
            <QuickInstall onMessage={setMsg} />
          </Section>
        ) : null}

        <SegmentedButtons
          value={mode}
          onValueChange={x => setMode(x as Mode)}
          buttons={[
            {value: 'record', label: 'Record', icon: 'microphone'},
            {value: 'file', label: 'File', icon: 'file-music-outline'},
            {value: 'text', label: 'Text', icon: 'text-to-speech', disabled: !settings.isEnabled('ttsVoice')},
            {value: 'live', label: 'Live', icon: 'broadcast', disabled: !settings.isEnabled('liveVoice')},
          ]}
        />

        {mode === 'record' ? (
          <Section title="Record & convert" subtitle="Speak for up to a few minutes, then convert" icon="microphone">
            <LevelMeter level={voice.recording ? voice.recordLevel : 0} />
            <View style={styles.row}>
              {voice.recording ? (
                <Button
                  mode="contained"
                  icon="stop"
                  buttonColor={theme.colors.error}
                  onPress={() => guard(() => voice.stopRecording())}
                  testID="voice-stop-rec"
                >
                  Stop
                </Button>
              ) : (
                <Button
                  mode="contained-tonal"
                  icon="record-circle-outline"
                  onPress={() => guard(() => voice.startRecording())}
                  disabled={voice.converting}
                  testID="voice-record"
                >
                  Record
                </Button>
              )}
              {voice.lastRecording ? (
                <>
                  <IconButton
                    icon="play"
                    accessibilityLabel="Play recording"
                    onPress={() => guard(() => voice.play(voice.lastRecording!))}
                  />
                  <Button
                    mode="contained"
                    icon="auto-fix"
                    disabled={!voice.ready || voice.converting}
                    onPress={() => guard(() => voice.convert(voice.lastRecording!, 'Recording'), 'Converted')}
                  >
                    Convert
                  </Button>
                </>
              ) : null}
            </View>
          </Section>
        ) : null}

        {mode === 'file' ? (
          <Section title="Convert an audio file" subtitle="WAV, MP3, M4A, OGG, FLAC… decoded natively" icon="file-music-outline">
            <Button
              mode="contained"
              icon="folder-music-outline"
              disabled={!voice.ready || voice.converting}
              onPress={() =>
                guard(async () => {
                  await ensureDirs();
                  const f = await pickAndImport(Paths.recordings, [], 'audio');
                  if (f) {
                    await voice.convert(f.path, f.name);
                  }
                }, 'Converted')
              }
            >
              Pick audio & convert
            </Button>
          </Section>
        ) : null}

        {mode === 'text' ? <TtsPanel onMessage={setMsg} /> : null}

        {mode === 'live' ? (
          <Section title="Live voice changer" subtitle={`Latency ≈ ${v?.chunkSeconds ?? 2.5}s chunk + compute`} icon="broadcast">
            <StatusChip severity={liveEval.status} />
            <IssueList issues={liveEval.issues} />
            {voice.live ? (
              <TileRow>
                <StatTile
                  label="RTF"
                  value={voice.live.realtimeFactor.toFixed(2)}
                  hint={voice.live.realtimeFactor < 1 ? 'keeping up' : 'too slow — raise chunk'}
                />
                <StatTile label="Latency" value={`${Math.round(voice.live.latencyMs + voice.live.lastChunkMs)} ms`} />
                <StatTile
                  label="Chunks"
                  value={String(voice.live.chunks)}
                  hint={voice.live.droppedBlocks ? `${voice.live.droppedBlocks} dropped` : undefined}
                />
              </TileRow>
            ) : null}
            <LevelMeter level={voice.live?.inputLevel ?? 0} />
            <LevelMeter level={voice.live?.outputLevel ?? 0} color={theme.colors.primary} />
            {voice.liveRunning ? (
              <Button mode="contained" icon="stop" buttonColor={theme.colors.error} onPress={() => guard(() => voice.stopLive())}>
                Stop live
              </Button>
            ) : (
              <Button
                mode="contained"
                icon="broadcast"
                disabled={!voice.ready || liveEval.status === 'block'}
                onPress={() => guard(() => voice.startLive())}
              >
                Start live
              </Button>
            )}
            <Text variant="bodySmall" style={{color: theme.colors.onSurfaceVariant}}>
              Use headphones to avoid feedback. Audio flows mic → C++ engine → speaker without touching JavaScript.
            </Text>
          </Section>
        ) : null}

        <Section
          title="Pitch"
          subtitle={`${(v?.pitchShift ?? 0) > 0 ? '+' : ''}${v?.pitchShift ?? 0} semitones · ${v?.pitchMethod?.toUpperCase()} tracker`}
          icon="tune-vertical"
        >
          <Slider
            minimumValue={-12}
            maximumValue={12}
            step={1}
            value={v?.pitchShift ?? 0}
            onSlidingComplete={x => voice.setPitchShift(x)}
            minimumTrackTintColor={theme.colors.primary}
            thumbTintColor={theme.colors.primary}
          />
          <View style={styles.row}>
            <Chip compact onPress={() => voice.setPitchShift(-12)}>
              −12 (F→M)
            </Chip>
            <Chip compact onPress={() => voice.setPitchShift(0)}>
              0
            </Chip>
            <Chip compact onPress={() => voice.setPitchShift(12)}>
              +12 (M→F)
            </Chip>
          </View>
        </Section>

        {voice.converting || voice.engineLoading ? (
          <Section
            title={voice.engineLoading ? 'Loading voice engine…' : `Converting… ${Math.round(voice.progress * 100)}%`}
            icon="cog-sync"
          >
            <ProgressBar progress={voice.engineLoading ? undefined : voice.progress} indeterminate={voice.engineLoading} />
            {voice.converting ? <Button onPress={() => voice.cancelConversion()}>Cancel</Button> : null}
          </Section>
        ) : null}

        {voice.engineInfo ? (
          <Section
            title="Engine"
            icon="chip"
            subtitle={`${voice.engineInfo.layout} · ${voice.engineInfo.sampleRate / 1000} kHz · index: off`}
          >
            <View style={styles.chipsWrap}>
              {voice.engineInfo.providers.map(p => (
                <Chip key={p} compact icon="flash">
                  {p}
                </Chip>
              ))}
            </View>
            {voice.engineInfo.warnings.map(w => (
              <Text key={w} variant="bodySmall" style={{color: theme.colors.onSurfaceVariant}}>
                ⓘ {w}
              </Text>
            ))}
            <Button onPress={() => voice.unloadEngine()}>Unload engine (free RAM)</Button>
          </Section>
        ) : null}

        <Section
          title="Results"
          icon="playlist-music"
          subtitle={voice.conversions.length ? 'Stored on this device' : 'Your conversions will appear here'}
        >
          {voice.conversions.map(c => (
            <List.Item
              key={c.id}
              title={`${c.voiceName} · ${c.inputLabel}`}
              description={`${formatDuration(c.stats.outputSeconds)} · ${c.stats.realtimeFactor.toFixed(2)}x RT · ${new Date(
                c.createdAt,
              ).toLocaleTimeString()}`}
              left={p => <List.Icon {...p} icon="waveform" />}
              right={() => (
                <View style={styles.row}>
                  {voice.playingPath === c.outputPath ? (
                    <IconButton icon="stop" accessibilityLabel="Stop" onPress={() => voice.stopPlayback()} />
                  ) : (
                    <IconButton icon="play" accessibilityLabel="Play" onPress={() => guard(() => voice.play(c.outputPath))} />
                  )}
                  <IconButton
                    icon="delete-outline"
                    accessibilityLabel="Delete"
                    onPress={() =>
                      Alert.alert('Delete result?', undefined, [
                        {text: 'Cancel'},
                        {text: 'Delete', style: 'destructive', onPress: () => voice.deleteConversion(c.id)},
                      ])
                    }
                  />
                </View>
              )}
            />
          ))}
        </Section>
      </Screen>
      <Snackbar visible={!!msg} onDismiss={() => setMsg(null)} duration={4000}>
        {msg}
      </Snackbar>
    </View>
  );
});

/** One-tap install of whatever the current settings are missing. */
const QuickInstall = observer(function QuickInstall({onMessage}: {onMessage: (m: string) => void}) {
  const {voice, settings} = useStores();
  const v = settings.voice;
  const enc = voicesCatalog.encoders.find(e => e.precision === (v?.encoderPrecision ?? 'int8')) ?? voicesCatalog.encoders[0];
  const method = v?.pitchMethod;
  const pitch =
    method === 'rmvpe' || method === 'fcpe'
      ? voicesCatalog.pitch.find(p => p.method === method && (method === 'fcpe' || p.precision === (v?.encoderPrecision ?? 'int8'))) ??
        voicesCatalog.pitch.find(p => p.method === method)
      : undefined;
  const firstVoice = voicesCatalog.voices[0];
  const theme = useTheme();
  const [step, setStep] = useState<{index: number; total: number; id: string; label: string} | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Everything still missing, in install order, so the user sees the whole plan and its size up front.
  const plan = [
    !voice.encoderAsset
      ? {id: enc.id, label: `Speech encoder · ${enc.name}`, bytes: enc.sizeBytes, run: () => voice.installAsset(enc, 'encoder')}
      : null,
    pitch && (method === 'rmvpe' || method === 'fcpe') && !voice.pitchAsset(method)
      ? {id: pitch.id, label: `Pitch tracker · ${pitch.name}`, bytes: pitch.sizeBytes, run: () => voice.installAsset(pitch, 'pitch')}
      : null,
    !voice.selectedVoice && firstVoice
      ? {
          id: firstVoice.id,
          label: `Voice · ${firstVoice.name}`,
          bytes: firstVoice.sizeBytes ?? 0,
          run: () => voice.installPresetVoice(firstVoice),
        }
      : null,
  ].filter(<T,>(x: T | null): x is T => x != null);
  const totalBytes = plan.reduce((a, p) => a + (p.bytes ?? 0), 0);
  const task = step ? voice.downloads.get(step.id) : undefined;

  const install = async () => {
    setError(null);
    const steps = [...plan];
    try {
      for (let i = 0; i < steps.length; i++) {
        setStep({index: i + 1, total: steps.length, id: steps[i].id, label: steps[i].label});
        await steps[i].run();
      }
      setStep(null);
      onMessage('Voice Studio is ready');
    } catch (e: any) {
      setStep(null);
      setError(e?.message ?? String(e));
    }
  };

  if (step) {
    return (
      <View style={styles.installBox} testID="voice-install-progress">
        <Text variant="labelLarge">
          Step {step.index} of {step.total} · {step.label}
        </Text>
        {task && task.state === 'downloading' ? (
          <DownloadProgress
            bytes={task.bytes}
            total={task.total}
            speedBps={task.speedBps}
            onCancel={() => voice.downloads.cancel(step.id)}
          />
        ) : (
          <View style={styles.row}>
            <ActivityIndicator size="small" />
            <Text variant="bodySmall">{task?.state === 'done' ? 'Verifying…' : 'Connecting…'}</Text>
          </View>
        )}
      </View>
    );
  }
  return (
    <View style={styles.installBox}>
      {error ? (
        <Text variant="bodySmall" style={{color: theme.colors.error}} testID="voice-install-error">
          {error}
        </Text>
      ) : null}
      <Button mode="contained" icon={error ? 'refresh' : 'download'} onPress={install} testID="voice-quick-install">
        {error ? 'Retry' : `Install recommended packs${totalBytes > 0 ? ` (${formatBytes(totalBytes)})` : ''}`}
      </Button>
    </View>
  );
});

const styles = StyleSheet.create({
  flex: {flex: 1},
  center: {alignItems: 'center', gap: spacing.md},
  chips: {gap: spacing.sm, paddingVertical: 4},
  chipsWrap: {flexDirection: 'row', flexWrap: 'wrap', gap: 6},
  row: {flexDirection: 'row', alignItems: 'center', gap: spacing.sm, flexWrap: 'wrap'},
  installBox: {gap: spacing.sm},
});
