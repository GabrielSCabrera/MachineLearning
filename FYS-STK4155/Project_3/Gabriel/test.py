from time import time
import numpy as np
import logging, os
import preprocess
import config
import sys

# Disable tensorflow warnings
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import SGD
import tensorflow as tf
import keras

gs_label            =   "dataset"
gs_kernel_size      =   (3,3)
gs_activation_hid   =   "relu"
gs_activation_out   =   "softmax"
gs_layers           =   [100,66]
gs_learning_rate    =   1E-3
gs_epochs           =   20
gs_batch_size       =   32
gs_filename         =   "EMNIST CNN Gridsearch"
gs_deploy_name      =   "EMNIST_CNN_Gridsearch_Deploy"

data = preprocess.read_data()
data = preprocess.one_hot(data)
# data = preprocess.combine(data)
data = preprocess.reshape_4D(data)

params = {"kernel_size"     :   gs_kernel_size,
          "activation_hid"  :   gs_activation_hid,
          "activation_out"  :   gs_activation_out,
          "layers"          :   gs_layers,
          "layers_out"      :   data["layers_out"],
          "learning_rate"   :   gs_learning_rate,
          "epochs"          :   gs_epochs,
          "batch_size"      :   gs_batch_size}

test_config = "config.json"
test_weights = "weights.h5"

X_train, y_train, X_test, y_test = \
data["train"]["X"], data["train"]["y"], data["test"]["X"], data["test"]["y"]

X_train, X_test = preprocess.scale_direct(X_train, X_test)

msg = "Requires cmdline arg 'load' or 'save'"
if len(sys.argv) == 2:
    if sys.argv[1].lower() == "load":

        with open(test_config) as json_file:
            json_config = json_file.read()
        CNN = keras.models.model_from_json(json_config)
        CNN.load_weights(test_weights)
        weights = CNN.get_weights()
        tot = 0
        for w in weights:
            tot += w.size
        print(tot)

        new_predictions = CNN.predict(X_test)
        maxima = np.argmax(new_predictions, axis = 1)
        expected = np.argmax(y_test, axis = 1)
        correct = (maxima == expected).astype(np.int64)
        correct = np.mean(correct)
        print(correct)

    elif sys.argv[1].lower() == "save":
        CNN = Sequential()
        for layer in params["layers"]:
            CNN.add(Conv2D(layer, kernel_size = params["kernel_size"],
                           activation = params["activation_hid"],
                           input_shape = config.input_shape))
            CNN.add(MaxPooling2D(pool_size = (2,2)))

        CNN.add(Flatten())
        CNN.add(Dense(params["layers_out"], activation = params["activation_out"]))

        optimizer = SGD(learning_rate = params["learning_rate"])
        CNN.compile(optimizer = optimizer, loss = config.loss,
                    metrics = config.metrics)

        out = CNN.fit(X_train, y_train, epochs = params["epochs"],
                      batch_size = params["batch_size"],
                      validation_data=[X_test, y_test], verbose = 1)

        json_config = CNN.to_json()
        with open(test_config, 'w') as json_file:
            json_file.write(json_config)

        CNN.save_weights(test_weights)
        CNN.summary()

    else:
        raise KeyError(msg)
else:
    raise KeyError(msg)
