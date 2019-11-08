# WMI_NN

Create virtual environment with virtualenv, then activate it
Install python libraries from requirements.txt

For keras, change the config file in: $HOME/.keras/keras.json
to have the following content:
```json
{
    "floatx": "float32",
    "epsilon": 1e-07,
    "backend": "mxnet",
    "image_data_format": "channels_first"
}
```

## Communication mechanism
**MQTT**  
A MQTT Broker needs to be running to test client.py and server.py.  
Download/install from [here](https://mosquitto.org/download/)

Suggested way of testing communication is:
1. Run Mosquitto Broker
2. Start server.py with a command like  
    ```bash 
    python server.py --name SERVER
    ```
3.  * Start client.py with a command like (for first *device*)
        ```bash 
        python server.py --name ORANGE
        ```
    * Start client.py with a command like (for second *device*)
        ```bash 
        python server.py --name APPLE
        ```

This will establish connection through server.py. The Server also starts a mini web-server in the runtime directory, in order to make file download possible. After both clients have connected the server tells them the filename of the neural network model, which they download from the web-server.