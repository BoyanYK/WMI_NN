import json

import numpy as np
from keras import datasets as datasets


def prepare_inputs():
    """Prepares inputs to the correct format and shape"""
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    # input image dimensions
    img_rows, img_cols = 28, 28

    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
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
    (x_train, y_train), (x_test, y_test) = prepare_inputs()
    print('Prepared inputs')
    inputs = x_train[:4] # ? Only getting 4 for dev purposes
    outputs = y_train[:4] # ? Only getting 4 for dev purposes

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