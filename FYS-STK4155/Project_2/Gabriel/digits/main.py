import matplotlib.pyplot as plt
from mnist import MNIST
from time import time
import pandas as pd
import numpy as np
import os, sys

sys.path.append("..")
from backend.neuralnet import NeuralNet, preprocess, upsample_binary, split

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
            arg = arg.strip().split("=")
            # arg = arg.lower().strip().split("=")
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

def one_hot_Y(Y):
    out = np.zeros((Y.shape[0], 10))
    axis0 = np.arange(0, Y.shape[0])
    idx = np.vstack([axis0, Y])
    out[idx[0], idx[1]] = 1
    return out

""" PROGRAM PARAMETERS """

# Size of each batch sent into the neural network
batchsize = 100
# Configuration of layers in the Neural Network
NN_layers = [int(784*(2/3)**x) for x in range(10)]
# Number of epochs, or total cycles over all batches
NN_epochs = 1000
# Learning Rate
learning_rate = 0.001
# Ridge Regularization Parameter
regularization_param = 1E-7
# Activation function
activation_fxn = "sigmoid"
# Activation function for output layer (None defaults to "activation_fxn")
output_activation_fxn = None
# Optimize for CUDA
GPU = True
# File in which to save the terminal output
terminal_output_file = "log.txt"
# Directory in which to save the terminal output; underscore allows for
# automatic numbering
dirname = "results_dig"
# If we want to load an older model, we can pass a string to the following
loadname = None
# How many test examples to display to screen
N_display = 10
# Random Seed
rand_seed = 112358

""" IMPORTING COMMAND-LINE ARGUMENT SETTINGS """

# Initializing command-line arguments; NB all keys must be lowercase!
all_args = {"save":[str, "dirname"], "load":[str, "loadname"],
            "display":[int, "N_display"], "epochs":[int, "NN_epochs"],
            "batchsize":[int, "batchsize"], "lr":[float,"learning_rate"],
            "reg":[float, "regularization_param"], "seed":[int, "rand_seed"],
            "activation":[str, "activation_fxn"],
            "activation_out":[str, "output_activation_fxn"],
            "GPU":[bool, "GPU"]}
parse_args(all_args)

np.random.seed(rand_seed)


""" READING THE DATA (using mnist) """

mndata = MNIST('.')
X_train, Y_train = mndata.load_training()
X_test, Y_test = mndata.load_testing()
X_train, Y_train = np.array(X_train), np.array(Y_train)
X_test, Y_test = np.array(X_test), np.array(Y_test)

""" PREPROCESSING """

X_train = ((X_train.T - np.mean(X_train, axis = 1).T)/np.std(X_train, axis = 1).T).T
X_test = ((X_test.T - np.mean(X_test, axis = 1).T)/np.std(X_test, axis = 1).T).T
Y_train = one_hot_Y(Y_train)
Y_test = one_hot_Y(Y_test)

""" INFORMATION FOR THE USER """

msg1 = (f"\nProcessed Dataset Dimensions:\n"
        f"\n\tX_train: (N= {X_train.shape[0]}, p= {X_train.shape[1]})"
        f"\n\tY_train: (M= {Y_train.shape[0]}, q= {Y_train.shape[1]})\n"
        f"\n\tX_test:  (N= {X_test.shape[0]}, p= {X_test.shape[1]})"
        f"\n\tY_test:  (M= {Y_test.shape[0]}, q= {Y_test.shape[1]})\n"
        f"\nNeural Network Parameters\n"
        f"\n\tBatch Size: {batchsize}\n\tLayer Configuration: {NN_layers}"
        f"\n\tEpochs: {NN_epochs}\n\tLearning Rate: {learning_rate:g}\n\t"
        f"Regularization Parameter: {regularization_param:g}\n\t"
        f"Random Seed: {rand_seed:d}\n\nActivation Functions\n\n\t"
        f"Hidden Layers: {activation_fxn}\n\tOutput Layer "
        f"{output_activation_fxn}\n\tCUDA: {str(GPU)}\n")
print(msg1)

""" IMPLEMENTING THE NEURAL NETWORK """

# Initializing the neural network
NN = NeuralNet()

""" TRAINING OR LOADING A MODEL, DEPENDING ON COMMAND-LINE ARGUMENTS """

if loadname is None:
    print("Training the Network:\n")

    # Inserting the training data into the network
    NN.set(X_train, Y_train)

    # Training the neural network with the parameters given earlier
    W,B = NN.train(epochs = NN_epochs, layers = NN_layers, lr = learning_rate,
    reg = regularization_param, batchsize = batchsize, GPU = GPU,
    activation_fxn = activation_fxn,
    output_activation_fxn = output_activation_fxn)
else:
    print(f"Loading from directory <{loadname}>\n")
    # Loading a previous neural network
    NN.load(loadname)

# Predicting outputs for the testing data
Y_predict = NN.predict(X_test)

""" ERROR ANALYSIS """

# Rounding the predicted values to zero and one
Y_predict[Y_predict >= 0.5] = 1
Y_predict[Y_predict < 1] = 0

# Displaying N example tests
img_dims = (28, 28)
idx = np.random.choice(Y_predict.shape[0], N_display)
for i in idx:
    predict = np.where(Y_predict[i] == 1)[0]
    expect = np.where(Y_test[i] == 1)[0]
    msg = (f"Expected = {expect};\tPredicted = {predict}")
    print(msg)
    plt.imshow(np.reshape(X_test[i,:], img_dims))
    plt.show()

# Calculating the elementwise difference in the predicted and expected outputs
correct = 0
incorrect = 0
for i,j in zip(Y_predict, Y_test):
    if np.array_equal(i, j):
        correct += 1
    else:
        incorrect += 1

# Displaying the total correct and incorrect outputs
total = correct + incorrect
msg2 = (f"\nTest Results\n\n\tNumber of incorrect outputs: {incorrect:.0f}/"
       f"{total:d}\n\tNumber of correct outputs: {correct:.0f}/{total:d}\n\t"
       f"Percent correct: {100*correct/total:.0f}%\n")
print(msg2)

NN.ROC(X_test, Y_test)

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

    NN_layers = np.concatenate([[X_train.shape[1]], NN_layers, [Y_train.shape[1]]])
    np.save(f"{dirname}/layers", NN_layers)

    with open(f"{dirname}/{terminal_output_file}", "w+") as outfile:
        outfile.write(msg1 + "\n" + msg2 + "\n")
