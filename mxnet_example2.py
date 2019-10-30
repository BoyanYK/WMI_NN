import keras
from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, InputLayer
from keras.optimizers import RMSprop
import numpy as np


def return_split_models(model, layer):
    model_f, model_h = Sequential(), Sequential()
    for current_layer in range(0, layer+1):
        model_f.add(model.layers[current_layer])
    # add input layer
    model_h.add(InputLayer(input_shape=model.layers[layer+1].input_shape[1:]))
    for current_layer in range(layer+1, len(model.layers)):
        model_h.add(model.layers[current_layer])
    return model_f, model_h

model = Sequential()
model.add(Dense(50,input_shape=(1,)))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model_f, model_h = return_split_models(model, 2)
print(model_f.summary())
print(model_h.summary())

print(model.predict([5]))

inter = model_f.predict([5])

print(model_h.predict(inter))






# IGNORE THE REST









# Create original model and save it
inputs = Input((1,))
dense_1 = Dense(10, activation='relu')(inputs)
dense_2 = Dense(10, activation='relu')(dense_1)
dense_3 = Dense(10, activation='relu')(dense_2)
dense_4 = Dense(10, activation='relu')(dense_3)
outputs = Dense(10)(dense_4)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=RMSprop(), loss='mse')
# model.summary()
model.save('test.h5')


# # Load the model and make modifications to it
model1 = load_model('test.h5')
model2 = load_model('test.h5')

for i in [0,1,2]:
    model2.layers.pop(0)
    model1.layers.pop()

# print(model1.summary())
# print(model2.summary())
# for layer in model1.layers:
#     print(layer)
    


# # Create your new model with the two layers removed and transfer weights
# out = Dense(10)(dense_2)
new_model1 = Model(inputs=inputs, outputs=model.layers[2].output)
new_model1.compile(optimizer=RMSprop(), loss='mse')
new_model1.set_weights(model1.get_weights())


# outX = new_model1.get_layer('dense_2').output

# x = Dense(3, activation='softmax')(outX)

# inp = Input(out)

# new_model2 = Model(inputs=model.layers[2].output, outputs=model.layers[5].output)
# new_model2.compile(optimizer=RMSprop(), loss='mse')
# new_model2.set_weights(model2.get_weights())



# inp = Input((10,))
# prev = inp
# cnt = 0
# lrs = []
# for lay in model2.layers:
#     lrs[cnt] = lay.    

# new_model2 = Model(inputs=inp, outputs=model.layers[5].output)
# new_model2.compile(optimizer=RMSprop(), loss='mse')
# new_model2.set_weights(model2.get_weights())


# new_model2.summary()
# new_model.save('test_complete.h5')