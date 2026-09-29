import Slider from '@react-native-community/slider';
import {observer} from 'mobx-react-lite';
import React, {useState} from 'react';
import {StyleSheet, View} from 'react-native';
import {Button, Chip, Dialog, Icon, Portal, RadioButton, Switch, Text, useTheme} from 'react-native-paper';

import {IssueList, Screen, Section, StatusChip} from '../components/ui';
import {FEATURES, FeatureDefinition, FeatureOption, VoiceSettings} from '../features/registry';
import {evaluate, Evaluation} from '../features/requirements';
import {RootScreenProps} from '../navigation/types';
import {useStores} from '../stores/RootStore';
import {spacing} from '../theme';

export const FeaturesScreen = observer(function FeaturesScreen({navigation}: RootScreenProps<'Features'>) {
  const {settings, device} = useStores();
  const [confirm, setConfirm] = useState<{f: FeatureDefinition; ev: Evaluation} | null>(null);

  const toggle = (f: FeatureDefinition, on: boolean) => {
    if (!on) {
      settings.setFeature(f.id, false);
      // Dependants go with it.
      FEATURES.filter(x => x.dependsOn?.includes(f.id)).forEach(x => settings.setFeature(x.id, false));
      return;
    }
    const ev = device.evaluation(f);
    if (ev.status === 'warn' || f.experimental) {
      setConfirm({f, ev});
      return;
    }
    enable(f);
  };

  const enable = (f: FeatureDefinition) => {
    f.dependsOn?.forEach(d => settings.setFeature(d, true));
    settings.setFeature(f.id, true);
    if (f.id === 'gpu') {
      settings.updateLlm({accel: 'gpu'});
    }
    if (f.id === 'npu') {
      settings.updateLlm({accel: 'npu'});
    }
    if (f.id === 'voice') {
      navigation.navigate('Main', {screen: 'Voice'});
    }
  };

  return (
    <View style={styles.flex}>
      <Screen testID="features-screen">
        <Text variant="bodyMedium" style={styles.intro}>
          MobiGPT is modular: heavy components are only downloaded when you enable them, and every feature is checked against this device's
          hardware first.
        </Text>
        {FEATURES.map(f => {
          const ev = device.evaluation(f);
          const on = settings.isEnabled(f.id);
          return (
            <Section
              key={f.id}
              title={f.title}
              subtitle={f.summary}
              icon={f.icon}
              testID={`feature-${f.id}`}
              right={
                f.core ? (
                  <Chip compact>Core</Chip>
                ) : (
                  <Switch
                    value={on}
                    disabled={ev.status === 'block' && !on}
                    onValueChange={v => toggle(f, v)}
                    testID={`feature-toggle-${f.id}`}
                  />
                )
              }
            >
              <View style={styles.row}>
                <StatusChip severity={ev.status} compact />
                {f.experimental ? (
                  <Chip compact icon="flask-outline">
                    Experimental
                  </Chip>
                ) : null}
                {f.footprint ? (
                  <Chip compact icon="download-outline">
                    {f.footprint}
                  </Chip>
                ) : null}
              </View>
              <IssueList issues={ev.issues} />
              {on && f.id === 'voice' && settings.voice ? f.options?.map(o => <OptionControl key={o.key} option={o} />) : null}
            </Section>
          );
        })}
      </Screen>
      <Portal>
        <Dialog visible={!!confirm} onDismiss={() => setConfirm(null)}>
          <Dialog.Icon icon="alert-outline" />
          <Dialog.Title>Enable {confirm?.f.title}?</Dialog.Title>
          <Dialog.Content style={styles.gap}>
            {confirm?.f.experimental ? (
              <Text variant="bodyMedium">This feature is experimental on phones and may be slower than CPU on some devices.</Text>
            ) : null}
            {confirm ? <IssueList issues={confirm.ev.issues} /> : null}
          </Dialog.Content>
          <Dialog.Actions>
            <Button onPress={() => setConfirm(null)}>Cancel</Button>
            <Button
              testID="feature-confirm"
              onPress={() => {
                if (confirm) {
                  enable(confirm.f);
                }
                setConfirm(null);
              }}
            >
              Enable anyway
            </Button>
          </Dialog.Actions>
        </Dialog>
      </Portal>
    </View>
  );
});

const OptionControl = observer(function OptionControl({option}: {option: FeatureOption}) {
  const theme = useTheme();
  const {settings, device} = useStores();
  const v = settings.voice!;
  const current = (v as unknown as Record<string, unknown>)[option.key];

  if (option.type === 'locked') {
    return (
      <View style={[styles.option, {backgroundColor: theme.colors.surfaceVariant}]}>
        <View style={styles.row}>
          <Icon source="lock-outline" size={18} color={theme.colors.primary} />
          <Text variant="titleSmall" style={styles.bold}>
            {option.label}
          </Text>
          <Chip compact>Always on</Chip>
        </View>
        <Text variant="bodySmall">{option.description}</Text>
        <Text variant="bodySmall" style={{color: theme.colors.onSurfaceVariant}}>
          {option.lockedReason}
        </Text>
      </View>
    );
  }

  if (option.type === 'slider') {
    return (
      <View style={[styles.option, {backgroundColor: theme.colors.surfaceVariant}]}>
        <Text variant="titleSmall" style={styles.bold}>
          {option.label}: {Number(current).toFixed(2)}
          {option.unit}
        </Text>
        <Text variant="bodySmall">{option.description}</Text>
        <Slider
          minimumValue={option.min}
          maximumValue={option.max}
          step={option.step}
          value={Number(current)}
          onSlidingComplete={x => settings.updateVoice({[option.key]: x} as Partial<VoiceSettings>)}
          minimumTrackTintColor={theme.colors.primary}
          thumbTintColor={theme.colors.primary}
        />
      </View>
    );
  }

  return (
    <View style={[styles.option, {backgroundColor: theme.colors.surfaceVariant}]}>
      <Text variant="titleSmall" style={styles.bold}>
        {option.label}
      </Text>
      <Text variant="bodySmall">{option.description}</Text>
      <RadioButton.Group value={String(current)} onValueChange={x => settings.updateVoice({[option.key]: x} as Partial<VoiceSettings>)}>
        {option.choices?.map(c => {
          const ev = c.requirement ? evaluate(c.requirement, device.profile) : null;
          return (
            <View key={c.value}>
              <RadioButton.Item
                label={c.label}
                value={c.value}
                disabled={ev?.status === 'block'}
                labelVariant="bodyMedium"
                style={styles.radio}
                testID={`option-${option.key}-${c.value}`}
              />
              {c.description ? (
                <Text variant="bodySmall" style={[styles.choiceDesc, {color: theme.colors.onSurfaceVariant}]}>
                  {c.description}
                </Text>
              ) : null}
              {ev && ev.status !== 'ok' ? (
                <View style={styles.choiceDesc}>
                  <IssueList issues={ev.issues} />
                </View>
              ) : null}
            </View>
          );
        })}
      </RadioButton.Group>
    </View>
  );
});

const styles = StyleSheet.create({
  flex: {flex: 1},
  intro: {opacity: 0.8},
  row: {flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', gap: 6},
  gap: {gap: spacing.sm},
  option: {borderRadius: 14, padding: spacing.md, gap: 4},
  bold: {fontWeight: '700'},
  radio: {paddingVertical: 2, paddingHorizontal: 0},
  choiceDesc: {marginLeft: 4, marginBottom: 4},
});
