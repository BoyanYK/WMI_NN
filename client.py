import argparse
import json
import pickle
import time
import urllib.request
import queue
import numpy as np
import time

import keras
from keras.applications.mobilenet import MobileNet
from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, InputLayer
from keras.optimizers import RMSprop

import paho.mqtt.client as mqtt

from threading import Thread

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="Device ID that must send the weather data",
                    action="store")
parser.add_argument("--host", help="Mosquitto host IP address", default='127.0.0.1',
                    action="store")
parser.add_argument("--port", help="Mosquitto host port", default=1884, type=int,
                    action="store")
args = parser.parse_args()

DEVICE_NAME = args.name
LOADED_MODEL = {}
data_queue = queue.Queue()
is_inferencing = False
model_split = {} # * Object that contains information on how neural network should be split

# TODO Move these functions to another file
def split_model_on(model, from_layer, to_layer):
    split_model = Sequential()
    if from_layer == 0:
        for current_layer in range(0, to_layer+1):
            split_model.add(model.layers[current_layer])
    elif to_layer == -1:
        split_model.add(InputLayer(input_shape=model.layers[from_layer+1].input_shape[1:]))
        for current_layer in range(from_layer+1, len(model.layers)):
            split_model.add(model.layers[current_layer])
    else:
        split_model.add(InputLayer(input_shape=model.layers[from_layer+1].input_shape[1:]))
        for current_layer in range(from_layer+1, to_layer+1):
            split_model.add(model.layers[current_layer])
    return split_model

def prepare_model(client, modelpath, model_split):
    global LOADED_MODEL
    print(DEVICE_NAME + " : attempting to load model")
    LOADED_MODEL = MobileNet()
    LOADED_MODEL.load_weights(modelpath)
    LOADED_MODEL = split_model_on(LOADED_MODEL, model_split[DEVICE_NAME]['layers_from'], model_split[DEVICE_NAME]['layers_to'])
    # LOADED_MODEL.summary()
    client.publish("devices/model_loaded", json.dumps({
        'from': DEVICE_NAME,
        'status': 'MODEL LOADED'
    }))
# TODO ^

def on_connect(client, userdata, flags, rc):
    """Handles initial connection to Mosquitto Broker"""
    if rc==0:
        print("connected OK Returned code=",rc)
    else:
        print("Bad connection Returned code=",rc)

def on_task(client, obj, msg):
    """
    Handler for receiving new data items in the message queue topic
    Adds them to a queue
    """
    task = json.loads(msg.payload)
    data_queue.put(task)
    is_inferencing = task['is_inferencing'] # ? Possibly unnecessary

def on_receive_model_info(client, obj, msg):
    """Handle downloading of neural network model"""
    data = json.loads(msg.payload)
    filename = data['filename']
    model_split = data['model_split']
    print(DEVICE_NAME + ' : ' + ' starting to download') # TODO Change this to better logging
    url = 'http://127.0.0.1:8000/' + data['filename']
    urllib.request.urlretrieve(url, DEVICE_NAME + '_model.h5')
    print(DEVICE_NAME + ' : ' + ' download complete') # TODO Change this to better logging
    prepare_model(client, DEVICE_NAME + '_model.h5', model_split)

def process_actions(client):
    """
    Main background thread loop to keep things going
    """
    global data_queue, is_inferencing
    while True:
        if data_queue.empty() == False or is_inferencing == True:
            task = data_queue.get()
            data = np.array(task['data'])
            result = LOADED_MODEL.predict(data)
            devices = task['for']
            # * Send output to next device
            if len(devices) != 0:
                recipient = devices[0]
                output = { 'data': result.tolist(), 'for': devices[1:], 'is_inferencing': True, 'started': task['started'] }
                client.publish(recipient + '/tasks', json.dumps(output))
            # * If last device, publish output
            else:
                output = {
                    'data': result.tolist(),
                    'started': task['started'],
                    'ended': time.time()
                }
                client.publish('output/results', json.dumps(output))
        else:
            time.sleep(0.01)

# * Register client
client = mqtt.Client(DEVICE_NAME)
# * Register callbacks
client.on_connect = on_connect
client.message_callback_add(DEVICE_NAME + "/tasks", on_task)
client.message_callback_add("init/models", on_receive_model_info)
# * Connect client
client.connect("127.0.0.1", port=1884)
time.sleep(2)
# * Subscribe to message topics
client.subscribe("devices/init")
client.subscribe("init/models")
client.subscribe(DEVICE_NAME + "/tasks")

# * Publish status
message = {
    'from': DEVICE_NAME,
    'status': 'on'
}
client.publish("devices/init", json.dumps(message)) #publish

# * Run main processing loop
process = Thread(target=process_actions, args=(client,))
process.start()
# * Main MQTT loop
client.loop_forever()