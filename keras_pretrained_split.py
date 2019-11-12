from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16, decode_predictions
from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, InputLayer
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import mxnet as mx
import numpy as np
import time

def return_split_models(model, layer):
    model_f, model_h = Sequential(), Sequential()
    for current_layer in range(0, layer+1):
        model_f.add(model.layers[current_layer])
    # add input layer
    model_h.add(InputLayer(input_shape=model.layers[layer+1].input_shape[1:]))
    for current_layer in range(layer+1, len(model.layers)):
        model_h.add(model.layers[current_layer])
    return model_f, model_h


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

model = VGG16(weights='imagenet')
print('VGG')
print(len(model.layers))

print(model.layers[0].input_shape[1:])
part1 = split_model_on(model, 0, 5)
print('PART 1')
print(len(part1.layers))
# part1.summary()

part2 = split_model_on(model, 5, 10)
print('PART 2')
print(len(part2.layers))
# part2.summary()

part3 = split_model_on(model, 10, 15)
print('PART 3')
print(len(part3.layers))
# part3.summary()

part4 = split_model_on(model, 15, -1)
print('PART 4')
print(len(part4.layers))
# part4.summary()
# print(model.layers[0].data_format)

img_path = 'dog.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
result = model.predict(x)

label = decode_predictions(result)
label = label[0][0]
# print the classification
print('%s (%.2f%%)' % (label[1], label[2]*100))


x = part1.predict(x)
x = part2.predict(x)
x = part3.predict(x)
x = part4.predict(x)

label = decode_predictions(x)
label = label[0][0]
# print the classification
print('%s (%.2f%%)' % (label[1], label[2]*100))