import argparse
import json
import pickle
import time
import urllib.request
import queue

import paho.mqtt.client as mqtt

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="Device ID that must send the weather data",
                    action="store")
parser.add_argument("--host", help="Mosquitto host IP address", default='127.0.0.1',
                    action="store")
parser.add_argument("--port", help="Mosquitto host port", default=1884, type=int,
                    action="store")                   
args = parser.parse_args()

DEVICE_NAME = args.name

data_queue = queue.Queue()
model_split = {} # * Object that contains information on how neural network should be split
# ? sample_model_split = {
# ?     '{DEVICE_NAME}' = {
# ?         layers_from = 3,
# ?         layers_to = 7,
# ?         output_receiver = '{DEVICE_NAME}'
# ?     } 
# ? }


def on_connect(client, userdata, flags, rc):
    """Handles initial connection to Mosquitto Broker"""
    if rc==0:
        print("connected OK Returned code=",rc)
    else:
        print("Bad connection Returned code=",rc)

def on_task(client, obj, msg):
    """
    Handler for receiving new data items in the message queue topic
    TODO Add them to an item queue that the model reads from
    """
    task = json.loads(msg.payload)
    data = task['data']
    target = task['for']
    result = 22 * data
    time.sleep(2)
    new_task = {
        'data': result,
        'for': 'output'
    }
    data_queue.put(new_task)
    # ? Likely to be deleted 
    # recipient = target + '/tasks'
    # if target == 'output':
    #     recipient = 'output/results'
    # client.publish(recipient, json.dumps(new_task))

def on_receive_model_info(client, obj, msg):
    """Handle downloading of neural network model"""
    data = json.loads(msg.payload)
    filename = data['filename']
    model_split = data['model_split'] # ! Not implemented on server side
    print(DEVICE_NAME + ' : ' + ' starting to download') # TODO Change this to better logging
    url = 'http://127.0.0.1:8000/' + data['filename']
    urllib.request.urlretrieve(url, DEVICE_NAME + '_model.h5')
    print(DEVICE_NAME + ' : ' + ' download complete') # TODO Change this to better logging

def run_inference(client, item):
    """
    Run inference and forward the results to the next device's message topic
    """
    time.sleep(2)
    recipient = model_split[DEVICE_NAME]['output_receiver'] + '/tasks'
    result = item['data'] # TODO Run actual prediction here
    client.publsih(recipient, json.dumps(output))

def process_actions(client):
    """
    Main background thread loop to keep things going
    """
    while True:
        if data_queue.empty() == False:
            run_inference(client, data_queue.get())
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
client.subscribe("devices/status")
client.subscribe("init/models")
client.subscribe(DEVICE_NAME + "/tasks")

# * Publish status
message = {
    'from': DEVICE_NAME,
    'status': 'on'
}
client.publish("devices/status", json.dumps(message)) #publish
client.loop_forever()