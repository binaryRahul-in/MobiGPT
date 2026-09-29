import {observer} from 'mobx-react-lite';
import React, {useState} from 'react';
import {Alert, FlatList, StyleSheet, View} from 'react-native';
import {Button, Card, Chip, FAB, Portal, ProgressBar, SegmentedButtons, Snackbar, Text, useTheme} from 'react-native-paper';

import {DownloadProgress, StatusChip} from '../components/ui';
import {TabScreenProps} from '../navigation/types';
import {ModelEntry} from '../stores/ModelStore';
import {useStores} from '../stores/RootStore';
import {spacing} from '../theme';
import {formatBytes, formatGB} from '../utils/format';

type Filter = 'recommended' | 'all' | 'downloaded';

export const ModelsScreen = observer(function ModelsScreen({navigation}: TabScreenProps<'Models'>) {
  const theme = useTheme();
  const {models, device} = useStores();
  const [filter, setFilter] = useState<Filter>('recommended');
  const [fabOpen, setFabOpen] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const data: ModelEntry[] =
    filter === 'downloaded'
      ? models.all.filter(m => m.localPath)
      : filter === 'recommended'
      ? [...models.all].sort((a, b) => rank(models, a) - rank(models, b))
      : models.all;

  const run = async (fn: () => Promise<unknown>, done?: string) => {
    try {
      await fn();
      if (done) {
        setMessage(done);
      }
    } catch (e: any) {
      setMessage(e?.message ?? String(e));
    }
  };

  const loadWithCheck = (m: ModelEntry) => {
    const c = models.compatibility(m);
    if (c.severity === 'block' || c.severity === 'warn') {
      Alert.alert(
        c.severity === 'block' ? 'Probably too large' : 'Heads up',
        `${m.name} needs about ${formatGB(c.requiredBytes)} of RAM; this device has ${formatGB(device.profile.totalRam)}.\n\n${c.message}`,
        [
          {text: 'Cancel', style: 'cancel'},
          {text: 'Load anyway', style: 'destructive', onPress: () => run(() => models.load(m.id, {force: true}), `${m.name} loaded`)},
        ],
      );
      return;
    }
    run(() => models.load(m.id), `${m.name} loaded`);
  };

  return (
    <View style={[styles.flex, {backgroundColor: theme.colors.background}]} testID="models-screen">
      <FlatList
        data={data}
        keyExtractor={m => m.id}
        contentContainerStyle={styles.list}
        ListHeaderComponent={
          <View style={styles.header}>
            <LoadedBanner />
            <SegmentedButtons
              value={filter}
              onValueChange={v => setFilter(v as Filter)}
              buttons={[
                {value: 'recommended', label: 'For you', icon: 'star-outline'},
                {value: 'all', label: 'All'},
                {value: 'downloaded', label: `On device (${models.downloaded.length})`},
              ]}
            />
          </View>
        }
        renderItem={({item}) => (
          <ModelCard
            m={item}
            onDownload={() => run(() => models.download(item.id), `${item.name} downloaded`)}
            onLoad={() => loadWithCheck(item)}
            onUnload={() => run(() => models.unload())}
            onDelete={() =>
              Alert.alert('Delete model?', `${item.name} (${formatBytes(item.sizeBytes)}) will be removed from this device.`, [
                {text: 'Cancel', style: 'cancel'},
                {text: 'Delete', style: 'destructive', onPress: () => run(() => models.remove(item.id))},
              ])
            }
          />
        )}
      />
      <Portal>
        <FAB.Group
          open={fabOpen}
          visible={navigation.isFocused()}
          icon={fabOpen ? 'close' : 'plus'}
          testID="models-fab"
          actions={[
            {icon: 'magnify', label: 'Browse Hugging Face', onPress: () => navigation.navigate('HFBrowser')},
            {
              icon: 'file-import-outline',
              label: 'Import GGUF from device',
              onPress: () => run(() => models.importFromDevice(), 'Model imported'),
            },
            {icon: 'speedometer', label: 'Benchmarks', onPress: () => navigation.navigate('Benchmark')},
          ]}
          onStateChange={({open}) => setFabOpen(open)}
        />
      </Portal>
      <Snackbar visible={!!message} onDismiss={() => setMessage(null)} duration={3500}>
        {message}
      </Snackbar>
    </View>
  );
});

function rank(models: ReturnType<typeof useStores>['models'], m: ModelEntry): number {
  const sev = {ok: 0, info: 0, warn: 1, block: 2}[models.compatibility(m).severity];
  // Fitting models first, then the most capable that still fit.
  return sev * 100 - (m.paramsB ?? 0);
}

const LoadedBanner = observer(function LoadedBanner() {
  const theme = useTheme();
  const {models, settings} = useStores();
  if (models.loading) {
    return (
      <Card mode="contained" style={[styles.banner, {backgroundColor: theme.colors.primaryContainer}]}>
        <Card.Content style={styles.gap}>
          <Text variant="titleSmall">Loading model… {Math.round(models.loadProgress * 100)}%</Text>
          <ProgressBar progress={models.loadProgress} />
        </Card.Content>
      </Card>
    );
  }
  if (!models.loaded) {
    return models.loadError ? (
      <Card mode="contained" style={[styles.banner, {backgroundColor: theme.colors.errorContainer}]}>
        <Card.Content>
          <Text variant="bodySmall">Last load failed: {models.loadError}</Text>
        </Card.Content>
      </Card>
    ) : null;
  }
  const info = models.loadedInfo;
  return (
    <Card mode="contained" style={[styles.banner, {backgroundColor: theme.colors.secondaryContainer}]} testID="loaded-banner">
      <Card.Title
        title={models.loaded.name}
        subtitle={`Loaded in ${(models.lastLoadMs / 1000).toFixed(1)}s · ctx ${settings.llm.nCtx}`}
        titleStyle={styles.bold}
      />
      <Card.Content style={styles.chips}>
        <Chip compact icon={info?.gpu ? 'expansion-card' : 'cpu-64-bit'}>
          {info?.gpu ? `GPU: ${(info.devices ?? []).join(', ') || 'on'}` : 'CPU'}
        </Chip>
        {info && !info.gpu && settings.llm.accel !== 'cpu' ? (
          <Chip compact icon="information-outline">
            {info.reasonNoGPU || 'GPU unavailable'}
          </Chip>
        ) : null}
        {info?.nParams ? <Chip compact>{(info.nParams / 1e9).toFixed(2)}B params</Chip> : null}
      </Card.Content>
      <Card.Actions>
        <Button onPress={() => models.unload()} testID="unload-model">
          Unload
        </Button>
      </Card.Actions>
    </Card>
  );
});

const ModelCard = observer(function ModelCard({
  m,
  onDownload,
  onLoad,
  onUnload,
  onDelete,
}: {
  m: ModelEntry;
  onDownload: () => void;
  onLoad: () => void;
  onUnload: () => void;
  onDelete: () => void;
}) {
  const theme = useTheme();
  const {models} = useStores();
  const task = models.downloads.get(m.id);
  const downloading = models.downloads.isActive(m.id);
  const compat = models.compatibility(m);
  const loaded = models.loadedId === m.id;
  return (
    <Card mode="contained" style={[styles.card, {backgroundColor: theme.colors.surface}]} testID={`model-${m.id}`}>
      <Card.Title
        title={m.name}
        titleStyle={styles.bold}
        subtitle={`${formatBytes(m.sizeBytes)} · ${models.quant(m)}${m.paramsB ? ` · ${m.paramsB}B` : ''}${
          m.license ? ` · ${m.license}` : ''
        }`}
        right={() => (
          <View style={styles.statusCol}>
            <StatusChip
              compact
              severity={compact(compat.severity)}
              label={compat.severity === 'block' ? 'Too large' : compat.severity === 'warn' ? 'Tight' : 'Fits'}
            />
          </View>
        )}
      />
      <Card.Content style={styles.gap}>
        {m.description ? (
          <Text variant="bodySmall" style={{color: theme.colors.onSurfaceVariant}}>
            {m.description}
          </Text>
        ) : null}
        <Text variant="labelSmall" style={{color: theme.colors.onSurfaceVariant}}>
          Needs ~{formatGB(compat.requiredBytes)} RAM at the current context size
        </Text>
        {downloading && task ? (
          <DownloadProgress bytes={task.bytes} total={task.total} speedBps={task.speedBps} onCancel={() => models.cancelDownload(m.id)} />
        ) : null}
        {task?.state === 'error' ? (
          <Text variant="bodySmall" style={{color: theme.colors.error}}>
            {task.error}
          </Text>
        ) : null}
      </Card.Content>
      <Card.Actions>
        {m.localPath ? (
          <>
            <Button onPress={onDelete} textColor={theme.colors.error} disabled={models.loading}>
              Delete
            </Button>
            {loaded ? (
              <Button mode="contained-tonal" onPress={onUnload}>
                Unload
              </Button>
            ) : (
              <Button mode="contained" onPress={onLoad} disabled={models.loading} testID={`load-${m.id}`}>
                Load
              </Button>
            )}
          </>
        ) : (
          <Button mode="contained" icon="download" onPress={onDownload} disabled={downloading || !m.repo} testID={`download-${m.id}`}>
            Download
          </Button>
        )}
      </Card.Actions>
    </Card>
  );
});

function compact(s: 'ok' | 'info' | 'warn' | 'block') {
  return s;
}

const styles = StyleSheet.create({
  flex: {flex: 1},
  list: {padding: spacing.md, gap: spacing.md, paddingBottom: 120},
  header: {gap: spacing.md},
  banner: {borderRadius: 20},
  card: {borderRadius: 20},
  bold: {fontWeight: '700'},
  gap: {gap: 6},
  chips: {flexDirection: 'row', flexWrap: 'wrap', gap: 6},
  statusCol: {paddingRight: spacing.md},
});
