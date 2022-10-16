import { StatusBar } from 'expo-status-bar';
import React from 'react';
import { Button, StyleSheet, Text, View } from 'react-native';


const Login = ({navigation}) => {

    const checkLogin = () => {
        navigation.navigate("MainScreen")
    }

    return (
        <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
            <Text>This is the Profile Screen!!!</Text>

            <Button 
                onPress={checkLogin}
                title='Login'
            />
        </View>
    )
}

export default Login

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
});
