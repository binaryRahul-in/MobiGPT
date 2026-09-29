import {observer} from 'mobx-react-lite';
import React from 'react';
import {RefreshControl, ScrollView, StyleSheet, View} from 'react-native';
import {Button, Chip, List, Text, useTheme} from 'react-native-paper';

import {Section, StatTile, StatusChip, TileRow} from '../components/ui';
import {recommendedMaxParamsB, TIER_LABEL} from '../features/device';
import {FEATURES, recommendedVoiceSettings} from '../features/registry';
import {TabScreenProps} from '../navigation/types';
import {useStores} from '../stores/RootStore';
import {brand, spacing} from '../theme';
import {formatBytes, formatGB} from '../utils/format';

export const DeviceScreen = observer(function DeviceScreen({navigation}: TabScreenProps<'Device'>) {
  const theme = useTheme();
  const {device, bench} = useStores();
  const p = device.profile;
  const rec = recommendedVoiceSettings(p);
  return (
    <ScrollView
      style={{backgroundColor: theme.colors.background}}
      contentContainerStyle={styles.content}
      refreshControl={<RefreshControl refreshing={device.probing} onRefresh={device.refresh} />}
      testID="device-screen"
    >
      <View style={[styles.hero, {backgroundColor: brand.violet}]}>
        <Text variant="labelLarge" style={styles.heroLabel}>
          {p.brand.toUpperCase()} · {p.platform === 'ios' ? 'iOS' : 'Android'} {p.osVersion}
          {p.isEmulator ? ' · emulator' : ''}
        </Text>
        <Text variant="headlineSmall" style={styles.heroTitle} testID="device-model">
          {p.model}
        </Text>
        <Text variant="bodyMedium" style={styles.heroLabel}>
          {TIER_LABEL[device.tier]} · runs Q4 models up to ~{recommendedMaxParamsB(device.tier)}B parameters
        </Text>
      </View>

      <TileRow>
        <StatTile label="RAM" value={formatGB(p.totalRam)} hint={`${formatGB(p.availableRam)} free now`} testID="device-ram" />
        <StatTile label="Storage free" value={formatBytes(p.freeStorage)} />
        <StatTile
          label="CPU"
          value={`${p.cores} cores`}
          hint={p.maxFreqMhz ? `up to ${(p.maxFreqMhz / 1000).toFixed(2)} GHz` : undefined}
        />
        <StatTile label="GPU" value={p.gpu.type || 'Unknown'} hint={p.gpu.name && p.gpu.name !== p.gpu.type ? p.gpu.name : undefined} />
      </TileRow>

      <Section title="Accelerators" icon="chip" subtitle="What llama.cpp and ONNX Runtime can use on this device">
        <List.Item
          title="CPU (NEON / AVX)"
          description={cpuFeatureSummary(p.cpuFeatures, p.hasDotProd, p.hasI8mm, p.hasFp16)}
          left={pp => <List.Icon {...pp} icon="cpu-64-bit" />}
          right={() => <StatusChip compact severity="ok" label="Always" />}
        />
        <List.Item
          title={`GPU · ${p.platform === 'ios' ? 'Metal' : 'OpenCL'}`}
          description={device.gpu.supported ? 'Layer offload available for chat models' : device.gpu.reason}
          descriptionNumberOfLines={3}
          left={pp => <List.Icon {...pp} icon="expansion-card" />}
          right={() => <StatusChip compact severity={device.gpu.supported ? 'ok' : 'block'} label={device.gpu.supported ? 'Yes' : 'No'} />}
        />
        <List.Item
          title={p.platform === 'ios' ? 'Neural Engine' : 'NPU · Hexagon HTP'}
          description={
            p.platform === 'ios'
              ? p.hasNpu
                ? 'Used by the voice engine through Core ML'
                : 'Not available'
              : device.npu.supported
              ? `Hexagon ${device.npu.generation} — experimental`
              : device.npu.reason
          }
          descriptionNumberOfLines={3}
          left={pp => <List.Icon {...pp} icon="brain" />}
          right={() => (
            <StatusChip
              compact
              severity={p.platform === 'ios' ? (p.hasNpu ? 'ok' : 'block') : device.npu.supported ? 'warn' : 'block'}
              label={(p.platform === 'ios' ? p.hasNpu : device.npu.supported) ? 'Yes' : 'No'}
            />
          )}
        />
        <Text variant="labelSmall" style={{color: theme.colors.onSurfaceVariant}}>
          llama.cpp devices
        </Text>
        <View style={styles.chips}>
          {(p.llamaDevices.length ? p.llamaDevices : ['CPU']).map(d => (
            <Chip key={d} compact>
              {d}
            </Chip>
          ))}
        </View>
        <Text variant="labelSmall" style={{color: theme.colors.onSurfaceVariant}}>
          ONNX Runtime providers (voice engine)
        </Text>
        <View style={styles.chips}>
          {p.voiceModuleAvailable ? (
            p.ortProviders.map(d => (
              <Chip key={d} compact>
                {d.replace('ExecutionProvider', '')}
              </Chip>
            ))
          ) : (
            <Chip compact icon="close">
              not in this build
            </Chip>
          )}
        </View>
      </Section>

      <Section
        title="Feature compatibility"
        icon="puzzle-outline"
        right={<Button onPress={() => navigation.navigate('Features')}>Manage</Button>}
      >
        {FEATURES.map(f => {
          const ev = device.evaluation(f);
          return (
            <List.Item
              key={f.id}
              title={f.title}
              description={ev.issues[0]?.message ?? f.summary}
              descriptionNumberOfLines={2}
              left={pp => <List.Icon {...pp} icon={f.icon} />}
              right={() => (
                <StatusChip compact severity={ev.status} label={ev.status === 'block' ? 'No' : ev.status === 'warn' ? 'Caveats' : 'Yes'} />
              )}
            />
          );
        })}
      </Section>

      <Section
        title="Recommended voice settings"
        icon="account-voice"
        subtitle="Applied automatically; change them under Features → Voice Studio"
      >
        <Text variant="bodyMedium">
          Pitch: {rec.pitchMethod.toUpperCase()} · Encoder: {rec.encoderPrecision.toUpperCase()} · Chunk: {rec.chunkSeconds}s · Loading:{' '}
          {rec.loadStrategy}
        </Text>
      </Section>

      <Section
        title="Benchmarks"
        icon="speedometer"
        subtitle={
          bench.best.tg
            ? `Best: ${bench.best.tg.toFixed(1)} tok/s generation · ${bench.best.pp.toFixed(0)} tok/s prompt`
            : 'Measure real speed on this phone'
        }
        right={
          <Button onPress={() => navigation.navigate('Benchmark')} testID="open-benchmarks">
            Open
          </Button>
        }
      />
    </ScrollView>
  );
});

function cpuFeatureSummary(features: string[], dot: boolean, i8mm: boolean, fp16: boolean): string {
  const flags = [
    dot && 'dotprod',
    i8mm && 'i8mm',
    fp16 && 'fp16',
    features.includes('sve') && 'sve',
    features.includes('avx2') && 'avx2',
  ].filter(Boolean);
  return flags.length ? `Fast paths: ${flags.join(', ')}` : 'Baseline SIMD';
}

const styles = StyleSheet.create({
  content: {padding: spacing.lg, gap: spacing.md, paddingBottom: spacing.xxl * 2},
  hero: {borderRadius: 24, padding: spacing.xl, gap: 4},
  heroTitle: {color: '#FFFFFF', fontWeight: '800'},
  heroLabel: {color: '#FFFFFFCC'},
  chips: {flexDirection: 'row', flexWrap: 'wrap', gap: 6},
});
