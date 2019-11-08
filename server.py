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
args = parser.parse_args()

devices = []
def on_connect(client, userdata, flags, rc):
    if rc==0:
        print("connected OK Returned code=",rc)
    else:
        print("Bad connection Returned code=",rc)

# def on_message(client, obj, msg):
#     if msg.topic == 'devices/status':
#         print(type(msg.payload))
#         device = str(msg.payload).replace(": ON", '')
#         print(device)
#         devices.append(device)
#     print(msg.topic + " " + str(msg.qos) + " " + str(msg.payload))

def on_status(client, obj, msg):
    pa = json.loads(msg.payload)
    device = pa['from']
    print(device)
    devices.append(device)
    if len(devices) == 2:
        # m = open('keras_mnist_cnn.h5')
        # model = m.read()
        # client.publish('init/models', bytes(model))
        task = {
            "filename": 'keras_mnist_cnn.h5',
            "for": devices[1]
        }
        client.publish('init/models', json.dumps(task))
        # client.publish(devices[0] + "/tasks", json.dumps(task))

def on_output(client, obj, msg):
    pa = json.loads(msg.payload)
    print('OUTPUT')
    print(pa['data'])

client = mqtt.Client(args.name)

sub_topic = args.name + '/'

client.on_connect = on_connect
# client.on_message = on_message
client.message_callback_add("devices/status", on_status)
client.message_callback_add("output/results", on_output)
client.connect("127.0.0.1", port=1884)

fServer = Thread(target=fileserver.start_server)
fServer.start()

client.publish("devices/status","SERVER: ON") #publish
client.subscribe("devices/status")




client.loop_forever()
time.sleep(2)

# client.loop_stop()
# rc = 0
# while rc == 0:
#     rc = client.loop()
# print("rc: " + str(rc))