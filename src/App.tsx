import {NavigationContainer} from '@react-navigation/native';
import {observer} from 'mobx-react-lite';
import React, {useEffect, useMemo, useState} from 'react';
import {ActivityIndicator, StatusBar, StyleSheet, useColorScheme, View} from 'react-native';
import {PaperProvider} from 'react-native-paper';
import {SafeAreaProvider} from 'react-native-safe-area-context';

import {Logo} from './brand/Logo';
import {RootNavigator} from './navigation/RootNavigator';
import {RootStore, StoreContext} from './stores/RootStore';
import {darkTheme, lightTheme, navigationTheme} from './theme';

const Themed = observer(function Themed({store}: {store: RootStore}) {
  const scheme = useColorScheme();
  const mode = store.settings.themeMode;
  const dark = mode === 'dark' || (mode === 'system' && scheme === 'dark');
  const paper = dark ? darkTheme : lightTheme;
  const nav = useMemo(() => navigationTheme(paper, dark), [paper, dark]);

  if (!store.settings.hydrated) {
    return (
      <View style={[styles.splash, {backgroundColor: paper.colors.background}]} testID="splash">
        <Logo size={88} />
        <ActivityIndicator color={paper.colors.primary} />
      </View>
    );
  }
  return (
    <PaperProvider theme={paper}>
      <StatusBar barStyle={dark ? 'light-content' : 'dark-content'} backgroundColor={paper.colors.background} />
      <NavigationContainer theme={nav}>
        <RootNavigator />
      </NavigationContainer>
    </PaperProvider>
  );
});

export default function App({store: injected}: {store?: RootStore}) {
  const [store] = useState(() => injected ?? new RootStore());
  useEffect(() => {
    store.bootstrap().catch(() => undefined);
  }, [store]);
  return (
    <SafeAreaProvider>
      <StoreContext.Provider value={store}>
        <Themed store={store} />
      </StoreContext.Provider>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  splash: {flex: 1, alignItems: 'center', justifyContent: 'center', gap: 24},
});
