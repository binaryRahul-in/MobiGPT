import {observer} from 'mobx-react-lite';
import React from 'react';
import {Linking, StyleSheet, View} from 'react-native';
import {Button, List, Text, useTheme} from 'react-native-paper';

import {Logo, Wordmark} from '../brand/Logo';
import {Screen, Section, StatusChip} from '../components/ui';
import {useStores} from '../stores/RootStore';
import {spacing} from '../theme';

const CREDITS: Array<[string, string, string]> = [
  ['llama.cpp / llama.rn', 'MIT', 'https://github.com/mybigday/llama.rn'],
  ['ONNX Runtime', 'MIT', 'https://github.com/microsoft/onnxruntime'],
  ['PocketPal AI (architecture inspiration, adapted helpers)', 'MIT', 'https://github.com/a-ghorbani/pocketpal-ai'],
  ['RVC — Retrieval-based Voice Conversion', 'MIT', 'https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI'],
  ['WORLD vocoder (DIO, Harvest, StoneMask)', 'BSD-3-Clause', 'https://github.com/mmorise/World'],
  ['FCPE pitch estimator', 'MIT', 'https://github.com/CNChTu/FCPE'],
  ['voiceclonnx (ONNX exports, INT8 study)', 'MIT', 'https://github.com/TigreGotico/voiceclonnx'],
  ['React Native Paper', 'MIT', 'https://github.com/callstack/react-native-paper'],
];

export const AboutScreen = observer(function AboutScreen() {
  const theme = useTheme();
  const {updates} = useStores();
  const r = updates.result;
  return (
    <Screen testID="about-screen">
      <View style={styles.brand}>
        <Logo size={96} />
        <Wordmark color={theme.colors.onBackground} />
        <Text variant="bodyMedium" style={{color: theme.colors.onSurfaceVariant}} testID="about-version">
          Version {updates.version}
        </Text>
      </View>

      <Section
        title="Updates"
        icon="update"
        subtitle={
          updates.checkedAt ? `Checked ${new Date(updates.checkedAt).toLocaleString()}` : 'Checks GitHub Releases — no other data is sent'
        }
      >
        {r?.updateAvailable && r.latest ? (
          <>
            <StatusChip severity="info" label={`Version ${r.latest.version} available`} />
            <Text variant="bodySmall" numberOfLines={12}>
              {r.latest.notes}
            </Text>
            <Button mode="contained" icon="download" onPress={() => Linking.openURL(r.latest!.apkUrl ?? r.latest!.url)}>
              Get the update
            </Button>
          </>
        ) : r ? (
          <StatusChip severity="ok" label="You're up to date" />
        ) : null}
        {updates.error ? <Text variant="bodySmall">Could not check: {updates.error}</Text> : null}
        <Button icon="refresh" loading={updates.checking} onPress={updates.check} testID="check-updates">
          Check now
        </Button>
      </Section>

      <Section title="Privacy" icon="shield-lock-outline">
        <Text variant="bodyMedium">
          Everything runs on this device. MobiGPT has no servers, accounts, analytics or crash reporting. Network access is used only to
          download models you choose and to check GitHub for updates.
        </Text>
      </Section>

      <Section title="Open source" icon="source-branch">
        <List.Item
          title="MobiGPT on GitHub"
          description="binaryRahul-in/MobiGPT"
          left={p => <List.Icon {...p} icon="github" />}
          onPress={() => Linking.openURL('https://github.com/binaryRahul-in/MobiGPT')}
        />
        {CREDITS.map(([name, license, url]) => (
          <List.Item key={name} title={name} description={license} onPress={() => Linking.openURL(url)} titleNumberOfLines={2} />
        ))}
      </Section>
    </Screen>
  );
});

const styles = StyleSheet.create({
  brand: {alignItems: 'center', gap: spacing.sm, paddingVertical: spacing.lg},
});
