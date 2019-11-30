from keras.applications.xception import Xception  # shite
from keras.applications.densenet import DenseNet121  # shite
from keras.applications.mobilenet import MobileNet  # works
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.vgg19 import VGG19
from keras.applications.nasnet import NASNetMobile
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

import datetime
import time
from keras.layers import InputLayer
from keras.models import load_model, Sequential


def split_model_on(model, from_layer, to_layer):
    split_model = Sequential()
    if from_layer == 0:
        for current_layer in range(0, to_layer + 1):
            split_model.add(model.layers[current_layer])
    elif to_layer == -1:
        split_model.add(InputLayer(input_shape=model.layers[from_layer + 1].input_shape[1:]))
        for current_layer in range(from_layer + 1, len(model.layers)):
            split_model.add(model.layers[current_layer])
    else:
        split_model.add(InputLayer(input_shape=model.layers[from_layer + 1].input_shape[1:]))
        for current_layer in range(from_layer + 1, to_layer + 1):
            split_model.add(model.layers[current_layer])
    return split_model


print("BEGIN DRY RUN")
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
print(st)

model = VGG19(include_top=True, weights='imagenet', input_tensor=None, classes=1000)
part1 = split_model_on(model, 0, 15)
part1.summary()
part2 = split_model_on(model, 16, -1)
part2.summary()
# image = load_img('dog.jpg', target_size=(299, 299))
# # convert the image pixels to a numpy array
# image = img_to_array(image)
#
# # reshape data for the model
# image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#
# # prepare the image for the VGG model
# image = preprocess_input(image)
#
# # predict the probability across all output classes
# yhat = model.predict(image)
#
# # convert the probabilities to class labels
# label = decode_predictions(yhat)
# # retrieve the most likely result, e.g. highest probability
# label = label[0][0]
# # print the classification
# print('%s (%.2f%%)' % (label[1], label[2] * 100))
# ts = time.time()
# st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
# print(st)
# print("END DRY RUN")
# count = 0
# y = 100
# for x in range(y):
#     ts = time.time()
#     st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
#     print("start run:" + str(x) + " " + st)
#     yhat = model.predict(image)
#
#     # convert the probabilities to class labels
#     label = decode_predictions(yhat)
#     # retrieve the most likely result, e.g. highest probability
#     label = label[0][0]
#     # print the classification
#     print('%s (%.2f%%)' % (label[1], label[2] * 100))
#     ts1 = time.time()
#     st1 = datetime.datetime.fromtimestamp(ts1).strftime('%Y-%m-%d %H:%M:%S')
#     tsA = ts1 - ts
#     print("end run:" + str(x) + " " + st1 + " total seconds: " + str(tsA))
#     print(" ")
#     count = count + tsA
#
# total = count
# count = count / y
# print("total runs: " + str(y))
# print("total time taken: " + str(total))
# print("average time taken: " + str(count))
