import paho.mqtt.client as mqtt
import argparse
import time
import json
import pickle
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="Device ID that must send the weather data",
                    action="store")
args = parser.parse_args()

def on_connect(client, userdata, flags, rc):
    if rc==0:
        print("connected OK Returned code=",rc)
    else:
        print("Bad connection Returned code=",rc)

def on_task(client, obj, msg):
    task = json.loads(msg.payload)
    data = task['data']
    target = task['for']
    result = 22 * data
    time.sleep(2)
    new_task = {
        'data': result,
        'for': 'output'
    }
    recipient = target + '/tasks'
    if target == 'output':
        recipient = 'output/results'
    client.publish(recipient, json.dumps(new_task))

def on_receive_model(client, obj, msg):
    data = json.loads(msg.payload)
    # data = pickle.loads(data)
    # model = data['data']
    # print(type(model))
    # with open(args.name + '.h5', 'wb') as fd:
    #     fd.write(msg.payload)
    # print('Beginning file download with wget module')
    print(args.name + ' : ' + ' starting to download')
    url = 'http://127.0.0.1:8000/' + data['filename']
    urllib.request.urlretrieve(url, args.name + '_model.h5')

    print(args.name + ' : ' + ' download complete')


client =mqtt.Client(args.name)

sub_topic = args.name + '/'

client.on_connect = on_connect
client.message_callback_add(args.name + "/tasks", on_task)
client.message_callback_add("init/models", on_receive_model)
client.connect("127.0.0.1", port=1884)
# client.loop_start()
time.sleep(2)
client.subscribe("devices/status")
client.subscribe("init/models")
client.subscribe(args.name + "/tasks")
message = {
    'from': args.name,
    'status': 'on'
}
client.publish("devices/status", json.dumps(message)) #publish
client.loop_forever()


# # client.loop_stop()
# rc = 0
# while rc == 0:
#     rc = mqtt.loop()
# print("rc: " + str(rc))