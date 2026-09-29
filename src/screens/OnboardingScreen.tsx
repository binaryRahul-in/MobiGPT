import {observer} from 'mobx-react-lite';
import React, {useRef, useState} from 'react';
import {FlatList, StyleSheet, useWindowDimensions, View} from 'react-native';
import {Button, Icon, Switch, Text, useTheme} from 'react-native-paper';
import {SafeAreaView} from 'react-native-safe-area-context';

import {Logo, Wordmark} from '../brand/Logo';
import {IssueList, StatusChip} from '../components/ui';
import {TIER_LABEL} from '../features/device';
import {FEATURES, FeatureId} from '../features/registry';
import {useStores} from '../stores/RootStore';
import {brand, spacing} from '../theme';
import {formatGB} from '../utils/format';

const PAGES = ['welcome', 'private', 'device', 'features'] as const;

const OPTIONAL: FeatureId[] = ['gpu', 'npu', 'voice', 'ttsVoice', 'liveVoice'];

export const OnboardingScreen = observer(function OnboardingScreen() {
  const theme = useTheme();
  const {width} = useWindowDimensions();
  const {settings, device} = useStores();
  const [page, setPage] = useState(0);
  const list = useRef<FlatList>(null);

  const go = (i: number) => {
    list.current?.scrollToIndex({index: i, animated: true});
    setPage(i);
  };

  const finish = () => settings.completeOnboarding();

  const renderPage = ({item}: {item: (typeof PAGES)[number]}) => {
    switch (item) {
      case 'welcome':
        return (
          <View style={[styles.page, {width}]} testID="onboarding-welcome">
            <Logo size={120} />
            <Wordmark color={theme.colors.onBackground} />
            <Text variant="titleMedium" style={[styles.center, {color: theme.colors.onSurfaceVariant}]}>
              Your private AI, running entirely on this phone.
            </Text>
            <View style={styles.bullets}>
              <Bullet icon="chat-processing-outline" text="Chat with open LLMs — Llama, Qwen, Gemma, Phi — powered by llama.cpp." />
              <Bullet icon="account-voice" text="Transform speech into any voice with on-device RVC." />
              <Bullet icon="chip" text="CPU, GPU and NPU acceleration picked for your hardware." />
            </View>
          </View>
        );
      case 'private':
        return (
          <View style={[styles.page, {width}]} testID="onboarding-private">
            <View style={[styles.hero, {backgroundColor: theme.colors.primaryContainer}]}>
              <Icon source="shield-lock-outline" size={64} color={theme.colors.onPrimaryContainer} />
            </View>
            <Text variant="headlineSmall" style={[styles.center, styles.bold]}>
              Offline by design
            </Text>
            <View style={styles.bullets}>
              <Bullet icon="cloud-off-outline" text="Prompts, recordings and voices never leave the device. No account, no telemetry." />
              <Bullet icon="download-outline" text="The network is only used when you download a model or check for updates." />
              <Bullet icon="puzzle-outline" text="Everything heavy is optional: install only the features you want." />
            </View>
          </View>
        );
      case 'device':
        return (
          <View style={[styles.page, {width}]} testID="onboarding-device">
            <View style={[styles.hero, {backgroundColor: theme.colors.secondaryContainer}]}>
              <Icon source="cellphone-cog" size={64} color={theme.colors.onSecondaryContainer} />
            </View>
            <Text variant="headlineSmall" style={[styles.center, styles.bold]}>
              {device.probed ? device.profile.model : 'Scanning your hardware…'}
            </Text>
            {device.probed ? (
              <View style={styles.bullets}>
                <Bullet icon="memory" text={`${formatGB(device.profile.totalRam)} RAM · ${TIER_LABEL[device.tier]}`} />
                <Bullet
                  icon="cpu-64-bit"
                  text={`${device.profile.cores} CPU cores${device.profile.hasDotProd ? ' · dotprod' : ''}${
                    device.profile.hasI8mm ? ' · i8mm' : ''
                  }`}
                />
                <Bullet
                  icon="expansion-card"
                  text={device.gpu.supported ? `GPU offload available (${device.profile.gpu.type})` : `GPU: ${device.gpu.reason ?? 'n/a'}`}
                />
                <Bullet
                  icon="chip"
                  text={
                    device.npu.supported ? `NPU: Hexagon ${device.npu.generation} (experimental)` : `NPU: ${device.npu.reason ?? 'n/a'}`
                  }
                />
              </View>
            ) : null}
            <Text variant="bodySmall" style={[styles.center, {color: theme.colors.onSurfaceVariant}]}>
              We use this to recommend models that fit and to warn you before installing anything too heavy.
            </Text>
          </View>
        );
      case 'features':
        return (
          <View style={[styles.page, styles.featuresPage, {width}]} testID="onboarding-features">
            <Text variant="headlineSmall" style={[styles.center, styles.bold]}>
              Pick your features
            </Text>
            <Text variant="bodySmall" style={[styles.center, {color: theme.colors.onSurfaceVariant}]}>
              Chat is always on. You can change these any time in Settings → Features.
            </Text>
            {FEATURES.filter(f => OPTIONAL.includes(f.id)).map(f => {
              const ev = device.evaluation(f);
              const blocked = ev.status === 'block';
              return (
                <View key={f.id} style={[styles.featureRow, {backgroundColor: theme.colors.surface}]}>
                  <View style={styles.flex}>
                    <View style={styles.rowGap}>
                      <Text variant="titleSmall" style={styles.bold}>
                        {f.title}
                      </Text>
                      <StatusChip severity={ev.status} compact label={blocked ? 'Unavailable' : ev.status === 'warn' ? 'Caveats' : 'OK'} />
                    </View>
                    <Text variant="bodySmall" style={{color: theme.colors.onSurfaceVariant}}>
                      {f.summary}
                    </Text>
                    {ev.issues.length ? <IssueList issues={ev.issues.slice(0, 2)} /> : null}
                  </View>
                  <Switch
                    testID={`onboarding-toggle-${f.id}`}
                    disabled={blocked}
                    value={settings.isEnabled(f.id)}
                    onValueChange={v => settings.setFeature(f.id, v)}
                  />
                </View>
              );
            })}
          </View>
        );
    }
  };

  return (
    <SafeAreaView style={[styles.flex, {backgroundColor: theme.colors.background}]} testID="onboarding">
      <FlatList
        ref={list}
        data={PAGES as unknown as (typeof PAGES)[number][]}
        keyExtractor={i => i}
        horizontal
        pagingEnabled
        showsHorizontalScrollIndicator={false}
        renderItem={renderPage}
        onMomentumScrollEnd={e => setPage(Math.round(e.nativeEvent.contentOffset.x / width))}
        getItemLayout={(_, index) => ({length: width, offset: width * index, index})}
      />
      <View style={styles.footer}>
        <View style={styles.dots}>
          {PAGES.map((p, i) => (
            <View
              key={p}
              style={[styles.dot, {backgroundColor: i === page ? brand.violet : theme.colors.outline, width: i === page ? 22 : 8}]}
            />
          ))}
        </View>
        <View style={styles.rowGap}>
          {page < PAGES.length - 1 ? (
            <>
              <Button onPress={finish} testID="onboarding-skip">
                Skip
              </Button>
              <Button mode="contained" onPress={() => go(page + 1)} testID="onboarding-next">
                Next
              </Button>
            </>
          ) : (
            <Button mode="contained" icon="rocket-launch-outline" onPress={finish} testID="onboarding-finish">
              Get started
            </Button>
          )}
        </View>
      </View>
    </SafeAreaView>
  );
});

function Bullet({icon, text}: {icon: string; text: string}) {
  const theme = useTheme();
  return (
    <View style={styles.bullet}>
      <Icon source={icon} size={22} color={theme.colors.primary} />
      <Text variant="bodyMedium" style={styles.flex}>
        {text}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  flex: {flex: 1},
  page: {flex: 1, alignItems: 'center', justifyContent: 'center', padding: spacing.xl, gap: spacing.lg},
  featuresPage: {justifyContent: 'flex-start', paddingTop: spacing.xxl, gap: spacing.sm},
  hero: {width: 128, height: 128, borderRadius: 40, alignItems: 'center', justifyContent: 'center'},
  center: {textAlign: 'center'},
  bold: {fontWeight: '700'},
  bullets: {alignSelf: 'stretch', gap: spacing.md, marginTop: spacing.sm},
  bullet: {flexDirection: 'row', gap: spacing.md, alignItems: 'center'},
  footer: {flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', padding: spacing.lg},
  dots: {flexDirection: 'row', gap: 6},
  dot: {height: 8, borderRadius: 4},
  rowGap: {flexDirection: 'row', alignItems: 'center', gap: spacing.sm},
  featureRow: {alignSelf: 'stretch', flexDirection: 'row', alignItems: 'center', gap: spacing.md, padding: spacing.md, borderRadius: 16},
});
