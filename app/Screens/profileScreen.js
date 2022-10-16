import { StatusBar } from 'expo-status-bar';
import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

export default function profile() {


  return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
          <Text>This is the Profile Screen!!!</Text>
      </View>
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
