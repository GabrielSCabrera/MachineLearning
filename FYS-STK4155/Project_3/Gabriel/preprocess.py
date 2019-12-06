# Suppressing FutureWarnings
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

import numpy as np
import config

from keras.utils import to_categorical

def read_data(N_max = None):
    # Reads the data as outlined in config.py
    data = {}
    for filename, label in zip(config.npy_names, config.files_labels):
        data[label] = {}
        data[label]["X"] = np.load(filename)[:N_max,1:]
        data[label]["y"] = np.load(filename)[:N_max,0,np.newaxis]
    return data

def one_hot(data):
    # Implementing one-hot encoding on 'y' in each data split
    layers_out = None
    for key in data.keys():
        data[key]["y"] = to_categorical(data[key]["y"])
        if layers_out is None:
            layers_out = data[key]["y"].shape[1]
        else:
            if layers_out != data[key]["y"].shape[1]:
                msg = ("Incompatible training and testing sets – different "
                       "one-hot configuration has occurred.")
                raise ValueError(msg)
    data["layers_out"] = layers_out
    return data

def reshape_4D(data, labels = config.files_labels):
    # Reshapes the data for compatibility with Keras' 4D datastructures
    for key in labels:
        step = data[key]["X"]
        shape = config.input_shape
        data[key]["X"] = step.reshape((step.shape[0], shape[0], shape[1], shape[2]))
    return data

def scale(data):
    # Scales the input arrays to zero mean and unit deviance
    scale = None
    shift = None
    for key in config.files_labels:
        data[key]["X"] = data[key]["X"].astype(np.float64)
        if scale is None:
            step = data[key]["X"]
            shift = np.mean(step)
            scale = np.var(step)
            data[key]["X"] = (step - shift)/scale
        else:
            data[key]["X"] = (data[key]["X"] - shift)/scale
    return data

def combine(data):
    """
        Combines the training and testing sets for X and y, respectively
        Intended to be used for grid search
    """
    X_train = data["train"]["X"]
    y_train = data["train"]["y"]
    X_test = data["test"]["X"]
    y_test = data["test"]["y"]
    X = np.concatenate([X_train, X_test], axis = 0)
    y = np.concatenate([y_train, y_test], axis = 0)
    data = {config.gs_label:{"X":X, "y":y}, "layers_out":data["layers_out"]}
    return data

if __name__ == "__main__":
    data = read_data()
    data = one_hot(data)
    data = scale(data)
    data = reshape_4D(data)
    print(data)
