from neuralnet import NeuralNet, preprocess, upsample_binary, split
from imageio import imread, imwrite
from time import time
import pandas as pd
import numpy as np
import os, sys

def parse_args(all_args):
    N_args = len(sys.argv)
    sp = 20
    valid_args = f"\n{'keywords':^{sp}s}{'type':^{sp}s}{'value':^{sp}s}\n"
    valid_args += "-"*len(valid_args) + "\n"
    valid_args = "\n\nValid options are as follows:\n" + valid_args
    for key, val in all_args.items():
        valid_args += f"{key:^{sp}s}{str(val[0]):^{sp}s}{val[1]:^{sp}s}\n"

    if N_args > 1:
        args = sys.argv[1:]
        prev_keys = []
        prev_vals = []
        prev_locs = []
        for arg in args:
            arg = arg.lower().strip().split("=")
            if len(arg) != 2:
                msg = (f"\n\nInvalid command-line argument format.  Keyword "
                       f"arguments are expected.  Attempted to pass {len(arg)}"
                       f" term(s) in one argument\n\t{arg}")
                msg += valid_args
                raise ValueError(msg)

            key = arg[0].strip()
            val = arg[1].strip()
            if key in prev_keys:
                msg = (f"\n\nAttempting to pass a command-line argument "
                       f"multiple times.\n\tkeyword:\t{key}\n\t  value:\t{val}")
                msg += valid_args
                raise NameError(msg)
            elif key in all_args.keys():
                new_type = all_args[key][0]
                msg = (f"\n\nAttempting to pass an argument of invalid "
                f"type.\n\tkeyword:\t{key}\n\t  value:\t{val}"
                f"\n\texpects:\t{all_args[key][0]}")
                msg += valid_args
                try:
                    val = new_type(val)
                except ValueError:
                    raise ValueError(msg)
                except TypeError:
                    raise TypeError(msg)
                prev_keys.append(key)
                prev_vals.append(val)
                prev_locs.append(all_args[key][1])
            else:
                msg = (f"\n\nAttempting to pass invalid command-line argument"
                       f"\n\tkeyword:\t{key}\n\t  value:\t{val}")
                msg += valid_args
                raise NameError(msg)
        for key,val,loc in zip(prev_keys, prev_vals, prev_locs):
            globals()[loc] = val

""" PROGRAM PARAMETERS """

# Size of each batch sent into the neural network
batchsize = 5
# Configuration of layers in the Neural Network
NN_layers = [100,150,100]
# Number of epochs, or total cycles over all batches
NN_epochs = 100
# File in which to save the terminal output
terminal_output_file = "term_out.txt"
# Directory in which to save the terminal output; underscore allows for
# automatic numbering
dirname = "image_"
# If we want to load an older model, we can pass a string to the following
loadname = None
# Save name for new image
imagename = "image_warp.jpg"

""" IMPORTING COMMAND-LINE ARGUMENT SETTINGS """

# Initializing command-line arguments; NB all keys must be lowercase!
all_args = {"dir":[str, "dirname"], "load":[str, "loadname"],
"img":[str, "imagename"]}
parse_args(all_args)

""" DATASET PARAMETERS """

# Determines which columns in the dataset are categorical, and which values
# the categories take

""" READING THE DATA (READING FROM .jpg FILES) """

X = imread("input.jpg")
Y = imread("output.jpg")

new_shape = (X.shape[0]*X.shape[1], X.shape[2])
old_shape = X.shape
X = np.reshape(X, new_shape)
Y = np.reshape(X, new_shape)

mean_X = np.mean(X)
std_X = np.std(X)

mean_Y = np.mean(Y)
std_Y = np.std(Y)

X = (X - mean_X)/std_X
Y = (Y - mean_Y)/std_Y

""" INFORMATION FOR THE USER """

msg1 = (f"\nProcessed Dataset Dimensions:\n"
        f"\n\tX: (N= {X.shape[0]}, p= {X.shape[1]})"
        f"\n\tY: (M= {Y.shape[0]}, q= {Y.shape[1]})\n"
        f"\nNeural Network Parameters\n"
        f"\n\tBatch Size: {batchsize}\n\tLayer Configuration: {NN_layers}"
        f"\n\tEpochs: {NN_epochs}\n")
print(msg1)

""" IMPLEMENTING THE NEURAL NETWORK """

# Initializing the neural network
NN = NeuralNet()

""" TRAINING OR LOADING A MODEL, DEPENDING ON COMMAND-LINE ARGUMENTS """

if loadname is None:
    print("Training the Network:\n")

    # Inserting the training data into the network
    NN.set(X, Y)

    # Training the neural network with the parameters given earlier
    W,B = NN.train(epochs = NN_epochs, layers = NN_layers, batchsize = batchsize)
else:
    print(f"Loading from directory <{loadname}>\n")
    # Loading a previous neural network
    NN.load(loadname)

Z = (NN.predict(Y)*std_Y) + mean_Y
Z = np.reshape(Z, old_shape).astype(np.uint8)
imwrite(imagename, Z)

""" SAVING DATA IF A NEW NETWORK IS CREATED"""

if loadname is None:
    if dirname[-1] == "_":
        ID = 0.0
        while True:
            ID_text = f"{ID:03.0f}"
            name_W = "W"
            name_B = "B"
            if os.path.isdir(dirname + ID_text):
                ID += 1
            else:
                dirname = dirname + ID_text
                os.mkdir(dirname)
                os.mkdir(dirname + "/W")
                os.mkdir(dirname + "/B")
                break
    elif not os.path.isdir(dirname):
        os.mkdir(dirname)
        os.mkdir(dirname + "/W")
        os.mkdir(dirname + "/B")

    for layer in range(len(W)):
        np.save(f"{dirname}/W/layer_{layer:03.0f}", W[layer])
        np.save(f"{dirname}/B/layer_{layer:03.0f}", B[layer])

    np.save(f"{dirname}/X", X)
    np.save(f"{dirname}/Y", Y)
    np.save(f"{dirname}/layers", NN_layers)

    with open(f"{dirname}/{terminal_output_file}", "w+") as outfile:
        outfile.write(msg1 + "\n")
