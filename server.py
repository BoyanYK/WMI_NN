import paho.mqtt.client as mqtt
import argparse
import time
import json

import fileserver
import server_model

from threading import Thread

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="Device ID that must send the weather data",
                    action="store")
parser.add_argument("--devices", help="Number of devices that are going to connect",
                    action="store", default=2, type=int)
args = parser.parse_args()

DEVICE_NAME = args.name
DEVICE_COUNT = args.devices

devices = []
loaded_parts = []

def on_connect(client, userdata, flags, rc):
    """Handles initial connection to Mosquitto Broker"""
    if rc==0:
        print("connected OK Returned code=",rc)
    else:
        print("Bad connection Returned code=",rc)

def on_model_loaded(client, obj, msg):
    """Confirm that all devices have loaded their part of the model successfully"""
    message = json.loads(msg.payload)
    print(message)
    loaded_parts.append(message['from'])
    if len(set(loaded_parts)) == DEVICE_COUNT:
        print('about to go to send inputs')
        inference = Thread(target=server_model.send_inputs, args=(client,devices))
        inference.start()

def on_init(client, obj, msg):
    """
    Deals with new connected devices. Once all devices have successfully connected
    execution can start
    """
    message = json.loads(msg.payload)
    device = message['from']
    devices.append(device)
    # * If all devices have connected tell them to download model
    if len(devices) == DEVICE_COUNT:
        task = {
            "filename": 'keras_mnist_cnn.h5',
            "model_split": {
                # TODO Not have this hardcoded
                devices[0]: {
                    "layers_from": 0,
                    "layers_to": 3,
                    "output_receiver": devices[1]
                },
                devices[1]: {
                    "layers_from": 3,
                    "layers_to": -1,
                    "output_receiver": 'output'
                } 
            }
        }
        print(task)
        client.publish('init/models', json.dumps(task))

def on_output(client, obj, msg):
    """Handle execution output"""
    result = json.loads(msg.payload)
    print('OUTPUT')
    print(result)

# * Initialise client and connect to broker
client = mqtt.Client(DEVICE_NAME)
client.on_connect = on_connect
# * Register event handlers for incoming messages
client.message_callback_add("devices/init", on_init)
client.message_callback_add("devices/model_loaded", on_model_loaded)
client.message_callback_add("output/results", on_output)
client.connect("127.0.0.1", port=1884)
# * Start a webserver to handle file downloads in a new thread
fServer = Thread(target=fileserver.start_server)
fServer.start()
# * Notify devices that the server is on and subscribe to interesting channels
client.publish("devices/init","SERVER: ON")
client.subscribe("devices/init")
client.subscribe("output/results")
client.subscribe("devices/model_loaded")
# * Keep MQTT client running in order to handle new incoming requests
client.loop_forever()