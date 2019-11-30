import argparse
import json
import pickle
import time
import urllib.request
import queue
import numpy as np
import time

import keras

from keras.models import Model, load_model, Sequential, model_from_json
from keras.layers import Input, Dense, InputLayer
from keras.optimizers import RMSprop
import tensorflow as tf
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
def split_model_on(model, starting_layer, end_layer):
    # create a new input layer for our sub-model we want to construct
    layer_name = model.layers[starting_layer].name
    new_input = Input(batch_shape=model.layers[starting_layer].get_input_shape_at(0))

    layer_outputs = {}

    def get_output_of_layer(layer):
        # if we have already applied this layer on its input(s) tensors,
        # just return its already computed output
        if layer.name in layer_outputs:
            return layer_outputs[layer.name]

        # if this is the starting layer, then apply it on the input tensor
        if layer.name == layer_name:
            out = layer(new_input)
            layer_outputs[layer.name] = out
            return out

        # find all the connected layers which this layer
        # consumes their output
        prev_layers = []
        for node in layer._inbound_nodes:
            prev_layers.extend(node.inbound_layers)

        # get the output of connected layers
        pl_outs = []
        for pl in prev_layers:
            pl_outs.extend([get_output_of_layer(pl)])

        # apply this layer on the collected outputs
        out = layer(pl_outs[0] if len(pl_outs) == 1 else pl_outs)
        layer_outputs[layer.name] = out
        return out

    # note that we start from the last layer of our desired sub-model.
    # this layer could be any layer of the original model as long as it is
    # reachable from the starting layer
    new_output = get_output_of_layer(model.layers[end_layer])

    # create the sub-model
    return Model(new_input, new_output)

def prepare_model(client, modelpath, model_split, model_name):
    global LOADED_MODEL, graph
    print(DEVICE_NAME + " : attempting to load model")
    # if "resnet" in model_name:
    #     from keras.applications.resnet50 import ResNet50
    #     LOADED_MODEL = ResNet50()
    # elif "mobile_net" in model_name:
    #     from keras.applications.mobilenet import MobileNet
    #     LOADED_MODEL = MobileNet()
    # LOADED_MODEL.load_weights(modelpath)
    # LOADED_MODEL._make_predict_function()
    # LOADED_MODEL = split_model_on(LOADED_MODEL, model_split[DEVICE_NAME]['layers_from'], model_split[DEVICE_NAME]['layers_to'])
    # graph = tf.get_default_graph()
    # LOADED_MODEL = load_model(modelpath)
    with open(DEVICE_NAME + '_model.json', 'r') as json_file:
        LOADED_MODEL = model_from_json(json_file.read())
    LOADED_MODEL.load_weights(DEVICE_NAME + "_model_weights.h5")
    graph = tf.get_default_graph()
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
    # filename = DEVICE_NAME + '.h5'
    model_split = data['model_split']
    print(DEVICE_NAME + ' : ' + ' starting to download') # TODO Change this to better logging
    url = 'http://127.0.0.1:8000/' + DEVICE_NAME + '.json'
    urllib.request.urlretrieve(url, DEVICE_NAME + '_model.json')
    print(DEVICE_NAME + ' : ' + ' model download complete') # TODO Change this to better logging
    url = 'http://127.0.0.1:8000/' + DEVICE_NAME + '_weights.h5'
    urllib.request.urlretrieve(url, DEVICE_NAME + '_model_weights.h5')
    print(DEVICE_NAME + ' : ' + ' weights download complete') # TODO Change this to better logging
    prepare_model(client, DEVICE_NAME + '_model.h5', model_split, data['filename'])

def process_actions(client):
    """
    Main background thread loop to keep things going
    """
    global data_queue, is_inferencing
    while True:
        if data_queue.empty() == False or is_inferencing == True:
            task = data_queue.get()
            data = np.array(task['data'])
            with graph.as_default():
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
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