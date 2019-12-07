from time import time
import numpy as np
import preprocess
import config

import logging, os

# Disable tensorflow warnings
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

import talos
from keras.layers import Dense, Conv2D, Flatten
from keras.models import Sequential, load_model
from keras.optimizers import SGD

def grid_search(X_train, y_train, X_test, y_test, params):
    """
        Creates a Convolutional Neural Network using the parameters given in
        argument 'params'
    """
    CNN = Sequential()
    for layer in params["layers"]:
        CNN.add(Conv2D(layer, kernel_size = params["kernel_size"],
                       activation = params["activation_hid"],
                       input_shape = config.input_shape))
    CNN.add(Flatten())
    CNN.add(Dense(params["layers_out"], activation = params["activation_out"]))

    optimizer = SGD(learning_rate = params["learning_rate"])
    CNN.compile(optimizer = optimizer, loss = config.loss,
                metrics = config.metrics)

    X_train, X_test = preprocess.scale_direct(X_train, X_test)

    out = CNN.fit(X_train, y_train, epochs = params["epochs"],
                  batch_size = params["batch_size"],
                  validation_data=[X_test, y_test], verbose = 2)

    return out, CNN

if __name__ == "__main__":

    t0 = time()

    data = preprocess.read_data()
    data = preprocess.one_hot(data)
    data = preprocess.combine(data)
    data = preprocess.reshape_4D(data, labels = [config.gs_label])

    params = {"kernel_size"     :   config.gs_kernel_size,
              "activation_hid"  :   config.gs_activation_hid,
              "activation_out"  :   config.gs_activation_out,
              "layers"          :   config.gs_layers,
              "layers_out"      :   [data["layers_out"]],
              "learning_rate"   :   config.gs_learning_rate,
              "epochs"          :   config.gs_epochs,
              "batch_size"      :   config.gs_batch_size}

    scan = talos.Scan(data[config.gs_label]["X"], data[config.gs_label]["y"],
                      model = grid_search, params = params,
                      experiment_name = config.gs_filename)

    scan.x = scan.x.reshape(scan.x.shape[0], scan.x.shape[1]*scan.x.shape[2])
    scan.y = scan.y.squeeze()

    talos.Deploy(scan, config.gs_deploy_name, metric = "categorical_accuracy")

    restored = talos.Restore(f"{config.gs_deploy_name}.zip")

    print(f"Time Elapsed: {time() - t0} seconds")
