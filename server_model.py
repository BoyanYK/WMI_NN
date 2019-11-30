import paho.mqtt.client as mqtt
from threading import Thread
import json
import time

import numpy as np

import keras
from keras.preprocessing import image
from keras import datasets as datasets
from keras import backend as K
from keras.models import Model, load_model, Sequential
from keras.layers import Input
from keras.applications.resnet50 import preprocess_input


def prepare_inputs():
    """Prepares inputs to the correct format and shape"""
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    # input image dimensions
    img_rows, img_cols = 224, 224

    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    return (x_train, y_train), (x_test, y_test)

def send_inputs(client, devices, iterations):
    """Loads in dataset input and sends it to the first device in the DDNN"""
    print('About to prepare inputs')
    # (x_train, y_train), (x_test, y_test) = prepare_inputs()
    print('Prepared inputs')
    # inputs = x_train[:4] # ? Only getting 4 for dev purposes
    # outputs = y_train[:4] # ? Only getting 4 for dev purposes

    img_path = 'dog.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # print('@@ EXPECTED OUTPUTS @@')
    # print(outputs)
    print('@@ ---------------- @@')
    device = devices[0]
    for i in range(iterations):
        a = np.array(x)
        task = { 
            'data': a.tolist(), 
            'for': devices[1:], # * List of recipients (aka 1 to last)
            'is_inferencing': True, 
            'started': time.time() 
        }
        client.publish(device + '/tasks', json.dumps(task))

def make_device_models(model, devices, device_config):
    if "resnet" in model:
        from keras.applications.resnet50 import ResNet50
        LOADED_MODEL = ResNet50()
    elif "mobile_net" in model:
        from keras.applications.mobilenet import MobileNet
        LOADED_MODEL = MobileNet()
    LOADED_MODEL.load_weights('{}.h5'.format(model))
    # LOADED_MODEL._make_predict_function()
    splits = []
    for dev in devices:
        split = split_model_on(LOADED_MODEL, device_config.pop(0), device_config.pop(0))
        with open('{}.json'.format(dev), 'w') as f:
            f.write(split.to_json())
        split.save_weights('{}_weights.h5'.format(dev))
    return True

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