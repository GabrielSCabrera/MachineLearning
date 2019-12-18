from time import time
import numpy as np
import preprocess
import logging
import config
import sys
import os

# Disable tensorflow warnings
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential, model_from_json
from keras.optimizers import SGD

def grid_element(data):
    """
        Creates a Convolutional Neural Network using the parameters given in
        argument 'params'
    """

    params = data["params"]

    metadata_filename = f"{config.gs_metadata_name}{data['counter']:04d}"
    with open(config.gs_directory + metadata_filename, "w+") as out:
        string = ""
        for key,val in data["params"].items():
            string += f"{key}\t{val}\n"
        out.write(string)

    X_train = data["data"]["train"]["X"]
    y_train = data["data"]["train"]["y"]
    X_test = data["data"]["test"]["X"]
    y_test = data["data"]["test"]["y"]

    weights_savename = f"{config.gs_weights_name}{data['counter']:04d}.h5"
    config_savename = f"{config.gs_config_name}{data['counter']:04d}.json"

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
    with open(config.gs_directory + config_savename, 'w') as json_file:
        json_file.write(json_config)

    CNN.save_weights(config.gs_directory + weights_savename)

def grid_search(data, params):
    combinations = []

    if not os.path.isdir(config.gs_directory):
        os.mkdir(config.gs_directory)

    keys = ["kernel_size", "activation_hid", "activation_out", "layers",
            "layers_out", "learning_rate", "epochs", "batch_size"]

    counter = -1
    for i1 in params["kernel_size"]:
        for i2 in params["activation_hid"]:
            for i3 in params["activation_out"]:
                for i4 in params["layers"]:
                    for i5 in params["layers_out"]:
                        for i6 in params["learning_rate"]:
                            for i7 in params["epochs"]:
                                for i8 in params["batch_size"]:
                                    counter += 1
                                    vals = [i1, i2, i3, i4, i5, i6, i7, i8]
                                    step = {"params":dict(zip(keys, vals))}
                                    step["data"] = data
                                    step["counter"] = counter
                                    combinations.append(step)
    for c in combinations:
        grid_element(c)

def load_grid(results_saved = True, subset = None):

    directory = config.gs_directory
    files = os.listdir(directory)

    if not results_saved:
        N = int(len(files)/3)
    else:
        N = int(len(files)/4)

    if subset is not None:
        N = subset

    data = []
    print(f"\rLOADING {0:>3d}%", end = "")
    for i in range(N):
        print(f"\rLOADING {int(100*i/N):>3d}%", end = "")
        weights_savename = f"{config.gs_weights_name}{i:04d}.h5"
        config_savename = f"{config.gs_config_name}{i:04d}.json"
        metadata_savename = f"{config.gs_metadata_name}{i:04d}"
        results_savename = f"{config.gs_results_name}{i+1:04d}"
        with open(directory + config_savename) as infile:
            model_config = infile.read()
        with open(directory + metadata_savename) as infile:
            metadata_list = infile.read().strip().split("\n")
            metadata = {}
            for m in metadata_list:
                dict_data = m.split("\t")
                metadata[dict_data[0].strip()] = dict_data[1].strip()
        if results_saved:
            with open(directory + results_savename) as infile:
                result = infile.read()
        CNN = model_from_json(model_config)
        CNN.load_weights(directory + weights_savename)
        data.append({"model":CNN, "metadata":metadata, "ID":i+1})
        if results_saved:
            data[-1]["result"] = float(result)
    print(f"\rLOADED  {100:>3d}%")
    return data

def grid_accuracies(models, data):

    X_train = data["train"]["X"]
    y_train = data["train"]["y"]
    X_test = data["test"]["X"]
    y_test = data["test"]["y"]

    for n,model in enumerate(models):
        CNN = model["model"]
        new_predictions = CNN.predict(X_test)
        maxima = np.argmax(new_predictions, axis = 1)
        expected = np.argmax(y_test, axis = 1)
        correct = (maxima == expected).astype(np.int64)
        model["result"] = np.mean(correct)
        print(f"Accuracy {n+1}/{len(models)}:\t{model['result']}")

    return models

def save_accuracies(models):
    for model in models:
        results_savename = f"{config.gs_results_name}{model['ID']:04d}"
        with open(config.gs_directory + results_savename, "w+") as outfile:
            outfile.write(str(model["result"]))

if __name__ == "__main__":

    t0 = time()

    data = preprocess.read_data()
    data = preprocess.one_hot(data)
    data = preprocess.reshape_4D(data)

    params = {"kernel_size"     :   config.gs_kernel_size,
              "activation_hid"  :   config.gs_activation_hid,
              "activation_out"  :   config.gs_activation_out,
              "layers"          :   config.gs_layers,
              "layers_out"      :   [data["layers_out"]],
              "learning_rate"   :   config.gs_learning_rate,
              "epochs"          :   config.gs_epochs,
              "batch_size"      :   config.gs_batch_size}

    msg = "Requires cmdline arg 'load' or 'save'"
    if len(sys.argv) == 2:
        if sys.argv[1].lower() == "load":
            models = load_grid()
        elif sys.argv[1].lower() == "save":
            grid_search(data, params)
            models = load_grid(False)
            models = grid_accuracies(models, data)
            save_accuracies(models)
        else:
            raise KeyError(msg)
    else:
        raise KeyError(msg)

    print(f"Time Elapsed: {time() - t0} seconds")
