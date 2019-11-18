import paho.mqtt.client as mqtt
import argparse
import time
import json

import fileserver
from threading import Thread

import numpy as np
import keras
from keras.datasets import mnist
from keras import backend as K
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
loaded_parts = []

def on_connect(client, userdata, flags, rc):
    """Handles initial connection to Mosquitto Broker"""
    if rc==0:
        print("connected OK Returned code=",rc)
    else:
        print("Bad connection Returned code=",rc)

def prepare_inputs():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # input image dimensions
    img_rows, img_cols = 28, 28

    # if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
    # else:
    #     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    #     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    #     input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    return (x_train, y_train), (x_test, y_test)

def on_model_loaded(client, obj, msg):
    """Confirm that all devices have loaded their part of the model successfully"""
    message = json.loads(msg.payload)
    print(message)
    loaded_parts.append(message['from'])
    if len(set(loaded_parts)) == DEVICE_COUNT:
        print('about to go to send inputs')
        inference = Thread(target=send_inputs, args=(client,))
        inference.start()

def send_inputs(client):
    print('About to prepare inputs')
    (x_train, y_train), (x_test, y_test) = prepare_inputs()
    print('Prepared inputs')
    inputs = x_train[:4]
    outputs = y_train[:4]

    print('@@ EXPECTED OUTPUTS @@')
    print(outputs)
    print('@@ ---------------- @@')
    device = devices[0]
    for image in inputs:
        image = np.array([image])
        task = {
            'data': image.tolist(),
            'for': devices[1:], # * List of recipients (aka 1 to last)
            'is_inferencing': True
        }
        client.publish(device + '/tasks', json.dumps(task))

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
                # TODO Per Device model splits
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