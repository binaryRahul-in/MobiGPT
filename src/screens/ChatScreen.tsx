import {observer} from 'mobx-react-lite';
import React, {useEffect, useRef, useState} from 'react';
import {FlatList, KeyboardAvoidingView, Platform, Pressable, StyleSheet, View} from 'react-native';
import {useHeaderHeight} from '@react-navigation/elements';
import {Button, Chip, IconButton, Snackbar, Text, TextInput, useTheme} from 'react-native-paper';

import {Logo} from '../brand/Logo';
import {EmptyState} from '../components/ui';
import {TabScreenProps} from '../navigation/types';
import {Message} from '../stores/ChatStore';
import {useStores} from '../stores/RootStore';
import {spacing} from '../theme';

const SUGGESTIONS = [
  'Explain how on-device AI protects my privacy',
  'Write a haiku about a phone that thinks',
  'Give me 3 tips to learn a language faster',
];

export const ChatScreen = observer(function ChatScreen({navigation}: TabScreenProps<'Chat'>) {
  const theme = useTheme();
  const headerHeight = useHeaderHeight();
  const {chat, models} = useStores();
  const [text, setText] = useState('');
  const [error, setError] = useState<string | null>(null);
  const listRef = useRef<FlatList<Message>>(null);
  const conv = chat.active;
  const messages = conv?.messages ?? [];

  useEffect(() => {
    navigation.setOptions({
      headerLeft: () => <IconButton icon="history" accessibilityLabel="Conversations" onPress={() => navigation.navigate('History')} />,
      headerTitle: () => (
        <Pressable onPress={() => navigation.navigate('Models')} style={styles.titleRow} testID="chat-model-chip">
          <Logo size={24} />
          <View>
            <Text variant="titleMedium" style={styles.bold}>
              MobiGPT
            </Text>
            <Text
              variant="labelSmall"
              style={{color: models.loaded ? theme.colors.secondary : theme.colors.onSurfaceVariant}}
              numberOfLines={1}
            >
              {models.loading ? `Loading… ${Math.round(models.loadProgress * 100)}%` : models.loaded?.name ?? 'No model loaded'}
            </Text>
          </View>
        </Pressable>
      ),
    });
  }, [navigation, models.loaded, models.loading, models.loadProgress, theme]);

  const send = async (value?: string) => {
    const t = (value ?? text).trim();
    if (!t) {
      return;
    }
    setText('');
    try {
      await chat.send(t);
    } catch (e: any) {
      setError(e?.message ?? String(e));
    }
  };

  if (!models.loadedId && !models.loading && messages.length === 0) {
    return (
      <View style={[styles.flex, {backgroundColor: theme.colors.background}]} testID="chat-empty">
        <EmptyState
          icon="cube-outline"
          title="Load a model to start"
          body={
            models.downloaded.length
              ? 'Pick one of your downloaded models.'
              : 'Download a model sized for your phone — everything runs locally.'
          }
          action={
            <Button mode="contained" icon="cube-outline" onPress={() => navigation.navigate('Models')} testID="chat-go-models">
              {models.downloaded.length ? 'Choose a model' : 'Browse models'}
            </Button>
          }
        />
      </View>
    );
  }

  return (
    <KeyboardAvoidingView
      style={[styles.flex, {backgroundColor: theme.colors.background}]}
      // Edge-to-edge Android ignores adjustResize, so both platforms pad above the keyboard.
      behavior="padding"
      keyboardVerticalOffset={headerHeight}
    >
      <FlatList
        ref={listRef}
        data={messages}
        keyExtractor={m => m.id}
        contentContainerStyle={styles.list}
        onContentSizeChange={() => listRef.current?.scrollToEnd({animated: true})}
        ListEmptyComponent={
          <View style={styles.suggestions}>
            <Logo size={56} />
            <Text variant="titleMedium" style={styles.bold}>
              What can I help with?
            </Text>
            {SUGGESTIONS.map(s => (
              <Chip key={s} icon="lightbulb-on-outline" onPress={() => send(s)} style={styles.suggestion}>
                {s}
              </Chip>
            ))}
          </View>
        }
        renderItem={({item}) => <Bubble m={item} streaming={chat.generating && item === messages[messages.length - 1]} />}
      />
      <View style={[styles.composer, {backgroundColor: theme.colors.surface, borderTopColor: theme.colors.outline}]}>
        <IconButton icon="plus" accessibilityLabel="New chat" onPress={() => chat.newConversation()} disabled={chat.generating} />
        <TextInput
          testID="chat-input"
          mode="outlined"
          dense
          multiline
          placeholder={models.loaded ? 'Message MobiGPT…' : 'Load a model first'}
          value={text}
          onChangeText={setText}
          style={styles.input}
          outlineStyle={styles.inputOutline}
          editable={!!models.loaded}
        />
        {chat.generating ? (
          <IconButton
            icon="stop-circle"
            iconColor={theme.colors.error}
            size={32}
            accessibilityLabel="Stop"
            onPress={() => chat.stop()}
            testID="chat-stop"
          />
        ) : (
          <IconButton
            icon="send-circle"
            iconColor={theme.colors.primary}
            size={32}
            accessibilityLabel="Send"
            disabled={!text.trim() || !models.loaded}
            onPress={() => send()}
            testID="chat-send"
          />
        )}
      </View>
      <Snackbar visible={!!error} onDismiss={() => setError(null)} duration={4000}>
        {error}
      </Snackbar>
    </KeyboardAvoidingView>
  );
});

const Bubble = observer(function Bubble({m, streaming}: {m: Message; streaming: boolean}) {
  const theme = useTheme();
  const [showReasoning, setShowReasoning] = useState(false);
  const mine = m.role === 'user';
  return (
    <View style={[styles.bubbleRow, mine ? styles.right : styles.left]}>
      <View
        style={[
          styles.bubble,
          mine
            ? {backgroundColor: theme.colors.primary, borderBottomRightRadius: 6}
            : {backgroundColor: theme.colors.surface, borderBottomLeftRadius: 6},
        ]}
      >
        {m.reasoning ? (
          <Pressable onPress={() => setShowReasoning(v => !v)}>
            <Text variant="labelSmall" style={{color: theme.colors.onSurfaceVariant}}>
              {showReasoning ? '▾ Thinking' : '▸ Thinking…'}
            </Text>
            {showReasoning ? (
              <Text variant="bodySmall" style={[styles.reasoning, {color: theme.colors.onSurfaceVariant}]}>
                {m.reasoning}
              </Text>
            ) : null}
          </Pressable>
        ) : null}
        <FormattedText text={m.content || (streaming ? '…' : '')} color={mine ? theme.colors.onPrimary : theme.colors.onSurface} />
        {m.error ? (
          <Text variant="bodySmall" style={{color: theme.colors.error}}>
            {m.error}
          </Text>
        ) : null}
        {m.stats && !mine ? (
          <Text variant="labelSmall" style={[styles.stats, {color: theme.colors.onSurfaceVariant}]}>
            {m.stats.tokens} tok · {m.stats.genTps.toFixed(1)} tok/s{m.stats.interrupted ? ' · stopped' : ''}
          </Text>
        ) : null}
      </View>
    </View>
  );
});

/** Lightweight markdown: fenced code blocks and **bold**; everything else as text. */
function FormattedText({text, color}: {text: string; color: string}) {
  const theme = useTheme();
  const parts = text.split(/```/);
  return (
    <View style={styles.formatted}>
      {parts.map((part, i) =>
        i % 2 === 1 ? (
          <Text key={i} selectable style={[styles.code, {backgroundColor: theme.colors.surfaceVariant, color: theme.colors.onSurface}]}>
            {part.replace(/^[a-zA-Z0-9_+-]*\n/, '').replace(/\n$/, '')}
          </Text>
        ) : part ? (
          <Text key={i} selectable variant="bodyMedium" style={{color}}>
            {part.split(/(\*\*[^*]+\*\*)/).map((seg, j) =>
              seg.startsWith('**') && seg.endsWith('**') ? (
                <Text key={j} style={styles.bold}>
                  {seg.slice(2, -2)}
                </Text>
              ) : (
                seg
              ),
            )}
          </Text>
        ) : null,
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  flex: {flex: 1},
  bold: {fontWeight: '700'},
  titleRow: {flexDirection: 'row', alignItems: 'center', gap: spacing.sm, maxWidth: 240},
  list: {padding: spacing.md, gap: spacing.sm, flexGrow: 1},
  suggestions: {flex: 1, alignItems: 'center', justifyContent: 'center', gap: spacing.md, paddingTop: spacing.xxl * 2},
  suggestion: {maxWidth: '100%'},
  bubbleRow: {flexDirection: 'row'},
  left: {justifyContent: 'flex-start'},
  right: {justifyContent: 'flex-end'},
  bubble: {maxWidth: '86%', borderRadius: 20, paddingHorizontal: 14, paddingVertical: 10, gap: 4},
  reasoning: {fontStyle: 'italic', marginVertical: 4},
  stats: {marginTop: 2},
  formatted: {gap: 6},
  code: {fontFamily: Platform.select({ios: 'Menlo', default: 'monospace'}), fontSize: 13, padding: 10, borderRadius: 10},
  composer: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    paddingHorizontal: 4,
    paddingVertical: 6,
    borderTopWidth: StyleSheet.hairlineWidth,
  },
  input: {flex: 1, maxHeight: 140},
  inputOutline: {borderRadius: 22},
});
