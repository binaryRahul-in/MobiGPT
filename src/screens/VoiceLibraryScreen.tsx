import {observer} from 'mobx-react-lite';
import React, {useState} from 'react';
import {Alert, StyleSheet, View} from 'react-native';
import {ActivityIndicator, Button, Chip, List, Searchbar, SegmentedButtons, Snackbar, Text, useTheme} from 'react-native-paper';

import {DownloadProgress, Screen, Section} from '../components/ui';
import {RootScreenProps} from '../navigation/types';
import {HFFile, HFModelSummary} from '../services/hf';
import {useStores} from '../stores/RootStore';
import {VoiceAssetPreset, voicesCatalog} from '../stores/VoiceStore';
import {spacing} from '../theme';
import {formatBytes} from '../utils/format';

type Tab = 'installed' | 'presets' | 'hub' | 'packs';

export const VoiceLibraryScreen = observer(function VoiceLibraryScreen({route}: RootScreenProps<'VoiceLibrary'>) {
  const theme = useTheme();
  const {voice} = useStores();
  const [tab, setTab] = useState<Tab>(route.params?.tab ?? 'installed');
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

  return (
    <View style={[styles.flex, {backgroundColor: theme.colors.background}]}>
      <SegmentedButtons
        style={styles.tabs}
        value={tab}
        onValueChange={t => setTab(t as Tab)}
        buttons={[
          {value: 'installed', label: 'Mine'},
          {value: 'presets', label: 'Presets'},
          {value: 'hub', label: 'Hub'},
          {value: 'packs', label: 'Engine'},
        ]}
      />
      <Screen>
        {tab === 'installed' ? (
          <Section
            title="Your voices"
            subtitle={`${voice.voices.length} installed · ${formatBytes(voice.installedBytes)} total`}
            icon="account-music"
          >
            <Button
              mode="contained"
              icon="file-import-outline"
              onPress={() => guard(() => voice.importVoice(), 'Voice imported')}
              testID="voice-import"
            >
              Import .onnx voice
            </Button>
            <Text variant="bodySmall" style={{color: theme.colors.onSurfaceVariant}}>
              Any RVC v1/v2 synthesiser exported to ONNX works (RVC WebUI "Export ONNX" or w-okada voice-changer). Have a .pth? Convert it
              on a PC with tools/rvc/export_voice_onnx.py.
            </Text>
            {voice.voices.map(v => (
              <List.Item
                key={v.id}
                title={v.name}
                description={`${v.version} · ${v.sampleRate / 1000} kHz · ${v.usesF0 ? 'f0' : 'no-f0'} · ${formatBytes(v.sizeBytes)} · ${
                  v.source
                }`}
                left={p => <List.Icon {...p} icon={v.id === voice.selectedVoice?.id ? 'check-circle' : 'account-voice'} />}
                onPress={() => voice.selectVoice(v.id)}
                right={() => (
                  <Button
                    textColor={theme.colors.error}
                    onPress={() =>
                      Alert.alert('Remove voice?', v.name, [
                        {text: 'Cancel'},
                        {text: 'Remove', style: 'destructive', onPress: () => voice.removeVoice(v.id)},
                      ])
                    }
                  >
                    Remove
                  </Button>
                )}
              />
            ))}
          </Section>
        ) : null}

        {tab === 'presets' ? (
          <Section title="Community voices" subtitle="Curated RVC voices hosted on Hugging Face" icon="star-outline">
            {voicesCatalog.voices.map(p => {
              const installed = voice.voices.some(v => v.id === p.id);
              const task = voice.downloads.get(p.id);
              const active = voice.downloads.isActive(p.id);
              return (
                <View key={p.id} style={styles.item}>
                  <List.Item
                    title={p.name}
                    description={`${p.description ?? ''}\n${p.repo} · ${p.license ?? ''}`}
                    descriptionNumberOfLines={3}
                    left={pp => <List.Icon {...pp} icon="account-voice" />}
                  />
                  {active && task ? (
                    <DownloadProgress
                      bytes={task.bytes}
                      total={task.total}
                      speedBps={task.speedBps}
                      onCancel={() => voice.downloads.cancel(p.id)}
                    />
                  ) : null}
                  <Button
                    mode={installed ? 'outlined' : 'contained-tonal'}
                    disabled={installed || active}
                    onPress={() => guard(() => voice.installPresetVoice(p), `${p.name} installed`)}
                  >
                    {installed ? 'Installed' : 'Install'}
                  </Button>
                </View>
              );
            })}
          </Section>
        ) : null}

        {tab === 'hub' ? <HubSearch onMessage={setMsg} /> : null}

        {tab === 'packs' ? (
          <>
            <PackList
              title="Content encoders"
              subtitle="HuBERT / ContentVec — turns speech into voice-independent features"
              kind="encoder"
              items={voicesCatalog.encoders}
              onMessage={setMsg}
            />
            <PackList
              title="Neural pitch trackers"
              subtitle="Optional: DSP trackers (Harvest, DIO, PM) need no download"
              kind="pitch"
              items={voicesCatalog.pitch}
              onMessage={setMsg}
            />
          </>
        ) : null}
      </Screen>
      <Snackbar visible={!!msg} onDismiss={() => setMsg(null)} duration={4500}>
        {msg}
      </Snackbar>
    </View>
  );
});

const PackList = observer(function PackList({
  title,
  subtitle,
  kind,
  items,
  onMessage,
}: {
  title: string;
  subtitle: string;
  kind: 'encoder' | 'pitch';
  items: VoiceAssetPreset[];
  onMessage: (m: string) => void;
}) {
  const {voice} = useStores();
  return (
    <Section title={title} subtitle={subtitle} icon={kind === 'encoder' ? 'waveform' : 'sine-wave'}>
      {items.map(a => {
        const installed = voice.isAssetInstalled(a.id);
        const task = voice.downloads.get(a.id);
        const active = voice.downloads.isActive(a.id);
        return (
          <View key={a.id} style={styles.item}>
            <List.Item
              title={a.name}
              description={`${a.precision.toUpperCase()} · ${formatBytes(a.sizeBytes)} · ${a.license ?? ''}${
                a.fidelity ? ` · fidelity ${(a.fidelity.cosMean * 100).toFixed(1)}% vs FP32` : ''
              }${a.notes ? `\n${a.notes}` : ''}`}
              descriptionNumberOfLines={4}
            />
            <View style={styles.row}>
              <Chip compact icon={a.precision === 'int8' ? 'lightning-bolt' : 'quality-high'}>
                {a.precision === 'int8' ? 'INT8 quantised' : 'Full precision'}
              </Chip>
            </View>
            {active && task ? (
              <DownloadProgress
                bytes={task.bytes}
                total={task.total}
                speedBps={task.speedBps}
                onCancel={() => voice.downloads.cancel(a.id)}
              />
            ) : null}
            {task?.state === 'error' ? <Text variant="bodySmall">{task.error}</Text> : null}
            {installed ? (
              <Button mode="outlined" onPress={() => voice.uninstallAsset(a.id)}>
                Uninstall
              </Button>
            ) : (
              <Button
                mode="contained-tonal"
                disabled={active}
                onPress={() =>
                  voice
                    .installAsset(a, kind)
                    .then(() => onMessage(`${a.name} installed`))
                    .catch(e => onMessage(e?.message ?? String(e)))
                }
              >
                Install
              </Button>
            )}
          </View>
        );
      })}
    </Section>
  );
});

const HubSearch = observer(function HubSearch({onMessage}: {onMessage: (m: string) => void}) {
  const {voice} = useStores();
  const [q, setQ] = useState('');
  const [busy, setBusy] = useState(false);
  const [results, setResults] = useState<HFModelSummary[]>([]);
  const [repo, setRepo] = useState<string | null>(null);
  const [files, setFiles] = useState<HFFile[]>([]);

  const search = async () => {
    setBusy(true);
    setRepo(null);
    try {
      setResults(await voice.searchHf(q));
    } catch (e: any) {
      onMessage(e?.message ?? String(e));
    } finally {
      setBusy(false);
    }
  };

  const open = async (id: string) => {
    setBusy(true);
    try {
      setFiles(await voice.listOnnx(id));
      setRepo(id);
    } catch (e: any) {
      onMessage(e?.message ?? String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Section title="Search Hugging Face" subtitle={voicesCatalog.search.hint} icon="magnify">
      <Searchbar placeholder="e.g. rvc onnx, singer name…" value={q} onChangeText={setQ} onSubmitEditing={search} onIconPress={search} />
      {busy ? <ActivityIndicator /> : null}
      {repo ? (
        <>
          <Button icon="arrow-left" onPress={() => setRepo(null)}>
            {repo}
          </Button>
          {files.length === 0 ? <Text variant="bodySmall">No .onnx files in this repository.</Text> : null}
          {files.map(f => (
            <List.Item
              key={f.path}
              title={f.path}
              description={formatBytes(f.size)}
              right={() => (
                <Button
                  compact
                  onPress={() =>
                    voice
                      .addHfVoice(repo, f.path)
                      .then(() => onMessage('Voice added'))
                      .catch(e => onMessage(e?.message ?? String(e)))
                  }
                >
                  Get
                </Button>
              )}
            />
          ))}
        </>
      ) : (
        results.map(r => (
          <List.Item
            key={r.id}
            title={r.id}
            description={`⬇ ${r.downloads} · ♥ ${r.likes}`}
            onPress={() => open(r.id)}
            left={p => <List.Icon {...p} icon="folder-outline" />}
          />
        ))
      )}
    </Section>
  );
});

const styles = StyleSheet.create({
  flex: {flex: 1},
  tabs: {margin: spacing.md, marginBottom: 0},
  item: {gap: 6, paddingBottom: spacing.sm},
  row: {flexDirection: 'row', gap: 6},
});
