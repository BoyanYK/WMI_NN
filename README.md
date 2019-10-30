# WMI_NN

Create virtual environment with virtualenv, then activate it
Install python libraries from requirements.txt

For keras, change the config file in: $HOME/.keras/keras.json
to have the following content:
```json
{
    "floatx": "float32",
    "epsilon": 1e-07,
    "backend": "mxnet",
    "image_data_format": "channels_first"
}
```