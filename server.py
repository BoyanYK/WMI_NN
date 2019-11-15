import paho.mqtt.client as mqtt
import argparse
import time
import json
import keras
import fileserver
from threading import Thread
from keras.models import Model, load_model, Sequential

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="Device ID that must send the weather data",
                    action="store")
parser.add_argument("--devices", help="Number of devices that are going to connect",
                    action="store", default=2, type=int)
args = parser.parse_args()

DEVICE_NAME = args.name
DEVICE_COUNT = args.devices

devices = []

def on_connect(client, userdata, flags, rc):
    """Handles initial connection to Mosquitto Broker"""
    if rc==0:
        print("connected OK Returned code=",rc)
    else:
        print("Bad connection Returned code=",rc)

def on_status(client, obj, msg):
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
                # TODO Per Device model splits
                # ? '{DEVICE_NAME}' = {
                # ?     layers_from = 3,
                # ?     layers_to = 7,
                # ?     output_receiver = '{DEVICE_NAME}'
                # ? } 
            },
            "for": devices[1]
        }
        client.publish('init/models', json.dumps(task))

def on_output(client, obj, msg):
    """Handle execution output"""
    result = json.loads(msg.payload)
    print('OUTPUT')
    print(result['data'])

# * Initialise client and connect to broker
client = mqtt.Client(DEVICE_NAME)
client.on_connect = on_connect
# * Register event handlers for incoming messages
client.message_callback_add("devices/status", on_status)
client.message_callback_add("output/results", on_output)
client.connect("127.0.0.1", port=1883)
# * Start a webserver to handle file downloads in a new thread
fServer = Thread(target=fileserver.start_server)
fServer.start()
# * Notify devices that the server is on and subscribe to interesting channels
client.publish("devices/status","SERVER: ON")
client.subscribe("devices/status")
client.subscribe("output/results")
# * Keep MQTT client running in order to handle new incoming requests
client.loop_forever()