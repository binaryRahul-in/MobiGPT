import {observer} from 'mobx-react-lite';
import React, {useState} from 'react';
import {StyleSheet, View} from 'react-native';
import {Button, IconButton, List, SegmentedButtons, Snackbar, Text, useTheme} from 'react-native-paper';

import {Screen, Section, StatTile, TileRow} from '../components/ui';
import {RootScreenProps} from '../navigation/types';
import {BENCH_PRESETS, BenchPreset} from '../stores/BenchmarkStore';
import {useStores} from '../stores/RootStore';
import {spacing} from '../theme';

export const BenchmarkScreen = observer(function BenchmarkScreen({navigation}: RootScreenProps<'Benchmark'>) {
  const theme = useTheme();
  const {bench, models, voice, settings} = useStores();
  const [preset, setPreset] = useState<BenchPreset>('quick');
  const [msg, setMsg] = useState<string | null>(null);
  const run = (fn: () => Promise<unknown>) => fn().catch(e => setMsg(e?.message ?? String(e)));
  const maxTg = Math.max(1, ...bench.results.map(r => r.tgTps ?? 0));

  return (
    <View style={styles.flex} testID="benchmark-screen">
      <Screen>
        <TileRow>
          <StatTile label="Best generation" value={bench.best.tg ? `${bench.best.tg.toFixed(1)} tok/s` : '—'} />
          <StatTile label="Best prompt" value={bench.best.pp ? `${bench.best.pp.toFixed(0)} tok/s` : '—'} />
          <StatTile
            label="Best voice RTF"
            value={bench.best.rtf ? `${bench.best.rtf.toFixed(2)}x` : '—'}
            hint="< 1 = faster than real time"
          />
        </TileRow>

        <Section
          title="LLM speed"
          icon="chat-processing-outline"
          subtitle={models.loaded ? `${models.loaded.name} · ${settings.llm.accel.toUpperCase()}` : 'Load a model first'}
        >
          <SegmentedButtons
            value={preset}
            onValueChange={v => setPreset(v as BenchPreset)}
            buttons={(Object.keys(BENCH_PRESETS) as BenchPreset[]).map(k => ({value: k, label: k[0].toUpperCase() + k.slice(1)}))}
          />
          <Text variant="bodySmall" style={{color: theme.colors.onSurfaceVariant}}>
            {BENCH_PRESETS[preset].label} — llama-bench methodology: prompt processing (pp) and token generation (tg) throughput.
          </Text>
          {models.loaded ? (
            <Button
              mode="contained"
              icon="play"
              loading={bench.running === 'llm'}
              disabled={!!bench.running}
              onPress={() => run(() => bench.runLlm(preset))}
              testID="bench-llm"
            >
              Run LLM benchmark
            </Button>
          ) : (
            <Button mode="contained-tonal" onPress={() => navigation.navigate('Main', {screen: 'Models'})}>
              Choose a model
            </Button>
          )}
        </Section>

        {settings.isEnabled('voice') ? (
          <Section
            title="Voice real-time factor"
            icon="account-voice"
            subtitle={
              voice.ready
                ? `${
                    voice.selectedVoice?.name
                  } · ${settings.voice?.pitchMethod?.toUpperCase()} · ${settings.voice?.encoderPrecision?.toUpperCase()}`
                : 'Install Voice Studio packs first'
            }
          >
            <Button
              mode="contained"
              icon="play"
              loading={bench.running === 'voice'}
              disabled={!!bench.running || !voice.ready}
              onPress={() => run(() => bench.runVoice(6))}
              testID="bench-voice"
            >
              Run voice benchmark (6 s audio)
            </Button>
          </Section>
        ) : null}

        <Section title="History" icon="history" right={bench.results.length ? <Button onPress={bench.clear}>Clear</Button> : undefined}>
          {bench.results.length === 0 ? <Text variant="bodySmall">No runs yet.</Text> : null}
          {bench.results.map(r => (
            <View key={r.id}>
              <List.Item
                title={r.kind === 'llm' ? `${r.subject}` : `Voice · ${r.subject}`}
                description={
                  r.kind === 'llm'
                    ? `pp ${r.ppTps?.toFixed(1)} · tg ${r.tgTps?.toFixed(1)} tok/s · ${r.accel} · ${r.threads} thr · ctx ${r.nCtx}`
                    : `RTF ${r.rtf?.toFixed(2)}x · enc ${r.encoderMs?.toFixed(0)} / f0 ${r.pitchMs?.toFixed(
                        0,
                      )} / synth ${r.synthMs?.toFixed(0)} ms · ${r.pitchMethod} · ${r.accel}`
                }
                descriptionNumberOfLines={2}
                left={p => <List.Icon {...p} icon={r.kind === 'llm' ? 'chat-processing-outline' : 'account-voice'} />}
                right={() => <IconButton icon="close" accessibilityLabel="Remove result" onPress={() => bench.remove(r.id)} />}
              />
              {r.kind === 'llm' && r.tgTps ? (
                <View style={[styles.bar, {backgroundColor: theme.colors.surfaceVariant}]}>
                  <View style={[styles.barFill, {width: `${(r.tgTps / maxTg) * 100}%`, backgroundColor: theme.colors.primary}]} />
                </View>
              ) : null}
            </View>
          ))}
        </Section>
      </Screen>
      <Snackbar visible={!!msg || !!bench.error} onDismiss={() => setMsg(null)}>
        {msg ?? bench.error}
      </Snackbar>
    </View>
  );
});

const styles = StyleSheet.create({
  flex: {flex: 1},
  bar: {height: 6, borderRadius: 3, marginHorizontal: spacing.lg, overflow: 'hidden'},
  barFill: {height: 6, borderRadius: 3},
});
