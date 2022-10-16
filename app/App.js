import { StatusBar } from 'expo-status-bar';
import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';

import profile from './Screens/profileScreen';
import settingsScreen from './Screens/settingsScreen';
import Login from './Screens/login';
import Register from './Screens/register';
import { createStackNavigator } from '@react-navigation/stack';

function ecgScreen () {
    return (
        <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
            <Text>This is the mainScreen!!!</Text>
        </View>
    )
}

function mainApp() {

  const Tab = createBottomTabNavigator();

  return (
      <Tab.Navigator>
          <Tab.Screen name="ECGScreen" component={ecgScreen} />
          <Tab.Screen name="Profile"  component={profile}/>
          <Tab.Screen name="Settings"  component={settingsScreen}/>
      </Tab.Navigator>  
  );
}



export default function StartScreens() {
  const Stack = createStackNavigator();
  return (
    <NavigationContainer>
        <Stack.Navigator initialRouteName='Login' screenOptions={{headerShown: false}}>
          {/* Add Welcome Screen */}
          <Stack.Screen name="Login" component={Login}/>
          <Stack.Screen name="Register" component={Register}/>
          <Stack.Screen name="MainScreen" component={mainApp}/>
        </Stack.Navigator>
    </NavigationContainer>
  )
}


const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
});
