# Suppressing FutureWarnings
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

from keras.layers import Dense, Conv2D, Flatten
from keras.models import Sequential, load_model
from keras.optimizers import SGD
import multiprocessing

import numpy as np
import preprocess
import config
import sys

def create_CNN(data):
    """
        Creates a Convolutional Neural Network using the configuration settings
        given in 'config.py'
    """
    CNN = Sequential()
    for layer in config.layers:
        CNN.add(Conv2D(layer, kernel_size = config.kernel_size,
                       activation = config.activation_hid,
                       input_shape = config.input_shape))
    CNN.add(Flatten())
    CNN.add(Dense(data["layers_out"], activation = config.activation_out))
    return CNN

def train_CNN(data, CNN):
    """
        Trains a CNN for a given training set of points.
    """
    X_train = data["train"]["X"]
    y_train = data["train"]["y"]
    X_test = data["test"]["X"]
    y_test = data["test"]["y"]

    optimizer = SGD(learning_rate = config.learning_rate)
    CNN.compile(optimizer = optimizer, loss = "categorical_crossentropy",
                metrics = config.metrics)
    CNN.fit(X_train, y_train, epochs = config.epochs,
            batch_size = config.batch_size)
    return CNN

def test_CNN(data, CNN):
    """
        Trains a CNN for a given training set of points.
    """
    X_test = data["test"]["X"]
    y_test = data["test"]["y"]
    evaluated = CNN.evaluate(X_test, y_test)
    results = {}
    for n,metric in enumerate(CNN.metrics_names):
        results[metric] = evaluated[n]
    return results

if __name__ == "__main__":
    data = preprocess.read_data()
    data = preprocess.one_hot(data)
    data = preprocess.scale(data)
    data = preprocess.reshape_4D(data)

    msg = "Requires cmdline arg 'load' or 'save'"
    if len(sys.argv) == 2:
        if sys.argv[1].lower() == "load":
            CNN = load_model(config.CNN_save_name)
        elif sys.argv[1].lower() == "save":
            CNN = create_CNN(data)
            CNN = train_CNN(data, CNN)
            CNN.save(config.CNN_save_name)
        else:
            raise KeyError(msg)
    else:
        raise KeyError(msg)

    CNN.summary()
    evaluation = test_CNN(data, CNN)
    print(evaluation)
