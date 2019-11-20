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

def send_inputs(client, devices):
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
    for i in range(10):
        a = np.array(x)
        task = { 
            'data': a.tolist(), 
            'for': devices[1:], # * List of recipients (aka 1 to last)
            'is_inferencing': True, 
            'started': time.time() 
        }
        client.publish(device + '/tasks', json.dumps(task))