import {observer} from 'mobx-react-lite';
import React, {useState} from 'react';
import {FlatList, StyleSheet, View} from 'react-native';
import {ActivityIndicator, Button, List, Searchbar, Snackbar, Text, useTheme} from 'react-native-paper';

import {StatusChip} from '../components/ui';
import {memoryFit} from '../features/requirements';
import {RootScreenProps} from '../navigation/types';
import {chatGgufFiles, HFFile, HFModelSummary, recommendFile} from '../services/hf';
import {useStores} from '../stores/RootStore';
import {spacing} from '../theme';
import {formatBytes} from '../utils/format';
import {estimateModelMemory} from '../utils/gguf';

export const HFBrowserScreen = observer(function HFBrowserScreen({navigation}: RootScreenProps<'HFBrowser'>) {
  const theme = useTheme();
  const {models, device, settings} = useStores();
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<HFModelSummary[]>([]);
  const [busy, setBusy] = useState(false);
  const [repo, setRepo] = useState<string | null>(null);
  const [files, setFiles] = useState<HFFile[]>([]);
  const [msg, setMsg] = useState<string | null>(null);

  const search = async () => {
    setBusy(true);
    setRepo(null);
    try {
      models.hf.setToken(settings.hfToken);
      setResults(await models.hf.search(query || 'instruct', 'gguf'));
    } catch (e: any) {
      setMsg(e?.message ?? String(e));
    } finally {
      setBusy(false);
    }
  };

  const open = async (id: string) => {
    setBusy(true);
    try {
      setRepo(id);
      setFiles(chatGgufFiles(await models.hf.listFiles(id)).sort((a, b) => a.size - b.size));
    } catch (e: any) {
      setMsg(e?.message ?? String(e));
      setRepo(null);
    } finally {
      setBusy(false);
    }
  };

  const add = async (f: HFFile) => {
    if (!repo) {
      return;
    }
    const id = models.addFromHf(repo, f);
    setMsg('Added — downloading…');
    try {
      await models.download(id);
      setMsg('Download complete');
      navigation.goBack();
    } catch (e: any) {
      setMsg(e?.message ?? String(e));
    }
  };

  const budget = device.profile.totalRam * 0.6;
  const best = repo ? recommendFile(files, budget) : undefined;

  return (
    <View style={[styles.flex, {backgroundColor: theme.colors.background}]}>
      <Searchbar
        placeholder="Search GGUF models (e.g. qwen, llama, phi)"
        value={query}
        onChangeText={setQuery}
        onSubmitEditing={search}
        onIconPress={search}
        style={styles.search}
        testID="hf-search"
      />
      {busy ? <ActivityIndicator style={styles.busy} /> : null}
      {repo ? (
        <FlatList
          data={files}
          keyExtractor={f => f.path}
          ListHeaderComponent={
            <View style={styles.repoHeader}>
              <Button icon="arrow-left" onPress={() => setRepo(null)}>
                Results
              </Button>
              <Text variant="titleMedium" style={styles.bold}>
                {repo}
              </Text>
              <Text variant="bodySmall" style={{color: theme.colors.onSurfaceVariant}}>
                Q4_K_M is the best size/quality trade-off on phones. ★ marks our pick for this device.
              </Text>
            </View>
          }
          renderItem={({item}) => {
            const need = estimateModelMemory({fileSizeBytes: item.size, nCtx: settings.llm.nCtx});
            const fit = memoryFit(need, device.profile);
            return (
              <List.Item
                title={`${item === best ? '★ ' : ''}${item.path}`}
                titleNumberOfLines={2}
                description={`${formatBytes(item.size)} · ${item.quant ?? ''}`}
                right={() => (
                  <View style={styles.fileRight}>
                    <StatusChip
                      compact
                      severity={fit.severity}
                      label={fit.severity === 'ok' ? 'Fits' : fit.severity === 'warn' ? 'Tight' : 'Too big'}
                    />
                    <Button compact mode="contained-tonal" onPress={() => add(item)}>
                      Get
                    </Button>
                  </View>
                )}
              />
            );
          }}
        />
      ) : (
        <FlatList
          data={results}
          keyExtractor={r => r.id}
          ListEmptyComponent={
            !busy ? (
              <Text style={styles.hint} variant="bodyMedium">
                Search the Hugging Face Hub for GGUF models. Gated models (e.g. official Llama) need a token in Settings.
              </Text>
            ) : null
          }
          renderItem={({item}) => (
            <List.Item
              title={item.id}
              description={`⬇ ${item.downloads.toLocaleString()} · ♥ ${item.likes}${item.gated ? ' · gated' : ''}`}
              left={p => <List.Icon {...p} icon="cube-outline" />}
              onPress={() => open(item.id)}
            />
          )}
        />
      )}
      <Snackbar visible={!!msg} onDismiss={() => setMsg(null)}>
        {msg}
      </Snackbar>
    </View>
  );
});

const styles = StyleSheet.create({
  flex: {flex: 1},
  search: {margin: spacing.md},
  busy: {margin: spacing.md},
  repoHeader: {padding: spacing.md, gap: 4},
  bold: {fontWeight: '700'},
  fileRight: {alignItems: 'flex-end', gap: 4, justifyContent: 'center'},
  hint: {padding: spacing.xl, textAlign: 'center', opacity: 0.7},
});
