import paho.mqtt.client as mqtt
import argparse
import time
import json
import time

import fileserver
import server_model

from threading import Thread

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="Device name",
                    action="store")
parser.add_argument("--model", help="Model name",
                    action="store", default='mobile_net')
parser.add_argument("--config", help="Config filepath",
                    action="store", default='configs.json')
parser.add_argument("--devices", help="Number of devices that are going to connect",
                    action="store", default=2, type=int)
args = parser.parse_args()

DEVICE_NAME = args.name
DEVICE_COUNT = args.devices
MODEL_NAME = args.model
with open(args.config) as json_file:
    data = json.load(json_file)
    CONFIG = data['models'][0][MODEL_NAME][str(DEVICE_COUNT)]

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
        model_split = {}
        for dev in range(DEVICE_COUNT - 1):
            model_split[devices[dev]] = {
                "layers_from": CONFIG.pop(0),
                "layers_to": CONFIG.pop(0),
                "output_receiver": devices[dev+1]
            }
            print(model_split)
        model_split[devices[DEVICE_COUNT - 1]] = {
            "layers_from": CONFIG.pop(0),
            "layers_to": CONFIG.pop(0),
            "output_receiver": "output"
        }
        print(model_split)
        task = {
            "filename": MODEL_NAME + '.h5',
            "model_split": model_split
        }
        print(task)
        client.publish('init/models', json.dumps(task))

def on_output(client, obj, msg):
    """Handle execution output"""
    ended = time.time()
    result = json.loads(msg.payload)
    started = result['started']
    device_ended = result['ended']
    print('#@#@#@#@#@ OUTPUT #@#@#@#@#@')
    print('started:' + str(started) + " ended: " + str(ended))
    print('Duration: ' + str(ended - started))
    print('Device Ended: ' + str(device_ended) + " Server ended: " + str(ended))
    print('Difference: ' + str(ended - device_ended))
    print('#@#@#@#@#@ END OF OUTPUT #@#@#@#@#@')


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