import React from 'react';
import {ScrollView, StyleSheet, View, ViewStyle} from 'react-native';
import {Button, Card, Chip, Icon, ProgressBar, Text, useTheme} from 'react-native-paper';
import {SafeAreaView} from 'react-native-safe-area-context';

import {Issue, Severity} from '../features/requirements';
import {severityColors, spacing} from '../theme';
import {formatBytes} from '../utils/format';

export function Screen({
  children,
  scroll = true,
  testID,
  style,
}: {
  children: React.ReactNode;
  scroll?: boolean;
  testID?: string;
  style?: ViewStyle;
}) {
  const theme = useTheme();
  const bg = {backgroundColor: theme.colors.background};
  if (!scroll) {
    return (
      <View style={[styles.flex, bg, style]} testID={testID}>
        {children}
      </View>
    );
  }
  return (
    <ScrollView
      style={[styles.flex, bg]}
      contentContainerStyle={[styles.content, style]}
      testID={testID}
      keyboardShouldPersistTaps="handled"
    >
      {children}
    </ScrollView>
  );
}

export function SafeScreen({children, testID}: {children: React.ReactNode; testID?: string}) {
  const theme = useTheme();
  return (
    <SafeAreaView style={[styles.flex, {backgroundColor: theme.colors.background}]} testID={testID}>
      {children}
    </SafeAreaView>
  );
}

export function Section({
  title,
  subtitle,
  icon,
  right,
  children,
  testID,
}: {
  title: string;
  subtitle?: string;
  icon?: string;
  right?: React.ReactNode;
  children?: React.ReactNode;
  testID?: string;
}) {
  const theme = useTheme();
  return (
    <Card mode="contained" style={[styles.card, {backgroundColor: theme.colors.surface}]} testID={testID}>
      <View style={styles.sectionHeader}>
        {icon ? (
          <View style={[styles.iconBubble, {backgroundColor: theme.colors.primaryContainer}]}>
            <Icon source={icon} size={20} color={theme.colors.onPrimaryContainer} />
          </View>
        ) : null}
        <View style={styles.flex}>
          <Text variant="titleMedium" style={styles.bold}>
            {title}
          </Text>
          {subtitle ? (
            <Text variant="bodySmall" style={{color: theme.colors.onSurfaceVariant}}>
              {subtitle}
            </Text>
          ) : null}
        </View>
        {right}
      </View>
      {children ? <View style={styles.sectionBody}>{children}</View> : null}
    </Card>
  );
}

export function StatTile({label, value, hint, testID}: {label: string; value: string; hint?: string; testID?: string}) {
  const theme = useTheme();
  return (
    <View style={[styles.tile, {backgroundColor: theme.colors.surfaceVariant}]} testID={testID}>
      <Text variant="labelSmall" style={{color: theme.colors.onSurfaceVariant}}>
        {label.toUpperCase()}
      </Text>
      <Text variant="titleMedium" style={styles.bold} numberOfLines={1}>
        {value}
      </Text>
      {hint ? (
        <Text variant="bodySmall" style={{color: theme.colors.onSurfaceVariant}} numberOfLines={2}>
          {hint}
        </Text>
      ) : null}
    </View>
  );
}

export function TileRow({children}: {children: React.ReactNode}) {
  return <View style={styles.tileRow}>{children}</View>;
}

const SEVERITY_LABEL: Record<Severity, string> = {ok: 'Ready', info: 'Ready', warn: 'Works with caveats', block: 'Not supported'};
const SEVERITY_ICON: Record<Severity, string> = {ok: 'check-circle', info: 'information', warn: 'alert', block: 'close-octagon'};

export function StatusChip({severity, label, compact, testID}: {severity: Severity; label?: string; compact?: boolean; testID?: string}) {
  const color = severityColors[severity];
  return (
    <Chip
      compact={compact}
      icon={SEVERITY_ICON[severity]}
      style={[styles.chip, {backgroundColor: `${color}22`}]}
      textStyle={{color, fontWeight: '600'}}
      testID={testID}
    >
      {label ?? SEVERITY_LABEL[severity]}
    </Chip>
  );
}

export function IssueList({issues}: {issues: Issue[]}) {
  if (!issues.length) {
    return null;
  }
  return (
    <View style={styles.issues}>
      {issues.map(i => (
        <View key={`${i.code}-${i.message}`} style={styles.issueRow}>
          <Icon source={SEVERITY_ICON[i.severity]} size={16} color={severityColors[i.severity]} />
          <Text variant="bodySmall" style={styles.issueText}>
            {i.message}
          </Text>
        </View>
      ))}
    </View>
  );
}

export function DownloadProgress({
  bytes,
  total,
  speedBps,
  onCancel,
}: {
  bytes: number;
  total: number;
  speedBps: number;
  onCancel?: () => void;
}) {
  const frac = total > 0 ? bytes / total : 0;
  return (
    <View style={styles.progress}>
      <ProgressBar progress={frac} style={styles.progressBar} />
      <View style={styles.progressRow}>
        <Text variant="bodySmall">
          {formatBytes(bytes)} / {formatBytes(total)} · {formatBytes(speedBps)}/s
        </Text>
        {onCancel ? (
          <Button compact onPress={onCancel}>
            Cancel
          </Button>
        ) : null}
      </View>
    </View>
  );
}

export function EmptyState({icon, title, body, action}: {icon: string; title: string; body: string; action?: React.ReactNode}) {
  const theme = useTheme();
  return (
    <View style={styles.empty}>
      <View style={[styles.emptyIcon, {backgroundColor: theme.colors.primaryContainer}]}>
        <Icon source={icon} size={36} color={theme.colors.onPrimaryContainer} />
      </View>
      <Text variant="titleLarge" style={[styles.bold, styles.center]}>
        {title}
      </Text>
      <Text variant="bodyMedium" style={[styles.center, {color: theme.colors.onSurfaceVariant}]}>
        {body}
      </Text>
      {action}
    </View>
  );
}

export function LevelMeter({level, color}: {level: number; color?: string}) {
  const theme = useTheme();
  const bars = 24;
  const lit = Math.round(Math.min(1, Math.sqrt(level)) * bars);
  return (
    <View style={styles.meter}>
      {Array.from({length: bars}).map((_, i) => (
        <View
          key={i}
          style={[
            styles.meterBar,
            {
              height: 6 + (i % 5) * 3 + (i < lit ? 10 : 0),
              backgroundColor: i < lit ? color ?? theme.colors.secondary : theme.colors.surfaceVariant,
            },
          ]}
        />
      ))}
    </View>
  );
}

export const styles = StyleSheet.create({
  flex: {flex: 1},
  content: {padding: spacing.lg, paddingBottom: spacing.xxl * 2, gap: spacing.md},
  card: {borderRadius: 20, overflow: 'hidden'},
  sectionHeader: {flexDirection: 'row', alignItems: 'center', gap: spacing.md, padding: spacing.lg, paddingBottom: spacing.sm},
  sectionBody: {paddingHorizontal: spacing.lg, paddingBottom: spacing.lg, gap: spacing.sm},
  iconBubble: {width: 36, height: 36, borderRadius: 12, alignItems: 'center', justifyContent: 'center'},
  bold: {fontWeight: '700'},
  center: {textAlign: 'center'},
  tile: {flexGrow: 1, flexBasis: '45%', padding: spacing.md, borderRadius: 16, gap: 2},
  tileRow: {flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm},
  chip: {alignSelf: 'flex-start'},
  issues: {gap: 6},
  issueRow: {flexDirection: 'row', gap: 8, alignItems: 'flex-start'},
  issueText: {flex: 1},
  progress: {gap: 4},
  progressBar: {height: 6, borderRadius: 3},
  progressRow: {flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center'},
  empty: {alignItems: 'center', padding: spacing.xl, gap: spacing.md},
  emptyIcon: {width: 72, height: 72, borderRadius: 24, alignItems: 'center', justifyContent: 'center'},
  meter: {flexDirection: 'row', alignItems: 'flex-end', gap: 3, height: 32},
  meterBar: {flex: 1, borderRadius: 2},
  row: {flexDirection: 'row', alignItems: 'center', gap: spacing.sm},
  wrap: {flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm},
});
