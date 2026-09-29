import {observer} from 'mobx-react-lite';
import React from 'react';
import {FlatList, StyleSheet, View} from 'react-native';
import {Button, IconButton, List, useTheme} from 'react-native-paper';

import {EmptyState} from '../components/ui';
import {RootScreenProps} from '../navigation/types';
import {useStores} from '../stores/RootStore';

export const HistoryScreen = observer(function HistoryScreen({navigation}: RootScreenProps<'History'>) {
  const theme = useTheme();
  const {chat} = useStores();
  return (
    <View style={[styles.flex, {backgroundColor: theme.colors.background}]}>
      <FlatList
        data={chat.sorted}
        keyExtractor={c => c.id}
        ListHeaderComponent={
          <Button
            icon="plus"
            mode="contained-tonal"
            style={styles.new}
            onPress={() => {
              chat.newConversation();
              navigation.goBack();
            }}
          >
            New conversation
          </Button>
        }
        ListEmptyComponent={
          <EmptyState icon="chat-outline" title="No conversations yet" body="Your chats are stored only on this device." />
        }
        renderItem={({item}) => (
          <List.Item
            title={item.title}
            description={`${item.messages.length} messages · ${item.modelName || 'no model'} · ${new Date(
              item.updatedAt,
            ).toLocaleString()}`}
            left={p => <List.Icon {...p} icon={item.id === chat.activeId ? 'chat' : 'chat-outline'} />}
            right={() => <IconButton icon="delete-outline" accessibilityLabel="Delete conversation" onPress={() => chat.remove(item.id)} />}
            onPress={() => {
              chat.select(item.id);
              navigation.goBack();
            }}
          />
        )}
      />
    </View>
  );
});

const styles = StyleSheet.create({
  flex: {flex: 1},
  new: {margin: 16},
});
