import { StatusBar } from 'expo-status-bar';
import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';

import profile from './profileScreen';
import settingsScreen from './settingsScreen';


function ecgScreen () {
    return (
        <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
            <Text>This is the mainScreen!!!</Text>
        </View>
    )
}

export default function mainApp() {

    const Tab = createBottomTabNavigator();

    return (
    <NavigationContainer>
        <Tab.Navigator>
            <Tab.Screen name="ECGScreen" component={ecgScreen} />
            <Tab.Screen name="Profile"  component={profile}/>
            <Tab.Screen name="Settings"  component={settingsScreen}/>
        </Tab.Navigator>
    </NavigationContainer>
    );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
});
