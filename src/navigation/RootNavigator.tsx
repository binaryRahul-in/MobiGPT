import {createBottomTabNavigator} from '@react-navigation/bottom-tabs';
import {useNavigation} from '@react-navigation/native';
import {createNativeStackNavigator, NativeStackNavigationProp} from '@react-navigation/native-stack';
import {observer} from 'mobx-react-lite';
import React from 'react';
import {Icon, IconButton, useTheme} from 'react-native-paper';

import {AboutScreen} from '../screens/AboutScreen';
import {BenchmarkScreen} from '../screens/BenchmarkScreen';
import {ChatScreen} from '../screens/ChatScreen';
import {DeviceScreen} from '../screens/DeviceScreen';
import {FeaturesScreen} from '../screens/FeaturesScreen';
import {HFBrowserScreen} from '../screens/HFBrowserScreen';
import {HistoryScreen} from '../screens/HistoryScreen';
import {ModelsScreen} from '../screens/ModelsScreen';
import {OnboardingScreen} from '../screens/OnboardingScreen';
import {SettingsScreen} from '../screens/SettingsScreen';
import {VoiceLibraryScreen} from '../screens/VoiceLibraryScreen';
import {VoiceScreen} from '../screens/VoiceScreen';
import {useStores} from '../stores/RootStore';
import {RootStackParamList, TabParamList} from './types';

const Stack = createNativeStackNavigator<RootStackParamList>();
const Tabs = createBottomTabNavigator<TabParamList>();

function SettingsButton() {
  const nav = useNavigation<NativeStackNavigationProp<RootStackParamList>>();
  return <IconButton icon="cog-outline" accessibilityLabel="Settings" testID="open-settings" onPress={() => nav.navigate('Settings')} />;
}

const TAB_ICONS: Record<keyof TabParamList, [string, string]> = {
  Chat: ['chat-processing', 'chat-processing-outline'],
  Models: ['cube', 'cube-outline'],
  Voice: ['account-voice', 'account-voice'],
  Device: ['cellphone-cog', 'cellphone-cog'],
};

const MainTabs = observer(function MainTabs() {
  const theme = useTheme();
  const {settings, device} = useStores();
  const voiceVisible = device.profile.voiceModuleAvailable || !device.probed;
  return (
    <Tabs.Navigator
      screenOptions={({route}) => ({
        headerRight: () => <SettingsButton />,
        headerTitleStyle: {fontWeight: '700'},
        tabBarActiveTintColor: theme.colors.primary,
        tabBarInactiveTintColor: theme.colors.onSurfaceVariant,
        tabBarStyle: {backgroundColor: theme.colors.surface, borderTopColor: theme.colors.outline},
        tabBarIcon: ({focused, color, size}) => <Icon source={TAB_ICONS[route.name][focused ? 0 : 1]} color={color} size={size} />,
        tabBarButtonTestID: `tab-${route.name.toLowerCase()}`,
      })}
    >
      <Tabs.Screen name="Chat" component={ChatScreen} options={{title: 'Chat'}} />
      <Tabs.Screen name="Models" component={ModelsScreen} />
      {voiceVisible ? (
        <Tabs.Screen
          name="Voice"
          component={VoiceScreen}
          options={{title: 'Voice Studio', tabBarLabel: 'Voice', tabBarBadge: settings.isEnabled('voice') ? undefined : 'new'}}
        />
      ) : null}
      <Tabs.Screen name="Device" component={DeviceScreen} />
    </Tabs.Navigator>
  );
});

export const RootNavigator = observer(function RootNavigator() {
  const {settings} = useStores();
  return (
    <Stack.Navigator screenOptions={{headerTitleStyle: {fontWeight: '700'}}}>
      {!settings.onboardingDone ? (
        <Stack.Screen name="Onboarding" component={OnboardingScreen} options={{headerShown: false}} />
      ) : (
        <>
          <Stack.Screen name="Main" component={MainTabs} options={{headerShown: false}} />
          <Stack.Screen name="HFBrowser" component={HFBrowserScreen} options={{title: 'Hugging Face'}} />
          <Stack.Screen name="VoiceLibrary" component={VoiceLibraryScreen} options={{title: 'Voice library'}} />
          <Stack.Screen name="Benchmark" component={BenchmarkScreen} options={{title: 'Benchmarks'}} />
          <Stack.Screen name="Features" component={FeaturesScreen} options={{title: 'Features'}} />
          <Stack.Screen name="Settings" component={SettingsScreen} />
          <Stack.Screen name="About" component={AboutScreen} options={{title: 'About & updates'}} />
          <Stack.Screen name="History" component={HistoryScreen} options={{title: 'Conversations', presentation: 'modal'}} />
        </>
      )}
    </Stack.Navigator>
  );
});
