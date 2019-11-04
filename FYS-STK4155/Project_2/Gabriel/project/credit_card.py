from time import time
import pandas as pd
import numpy as np
import sys
import os

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
# Percentage of data to set aside for testing
test_percent = 25
# Configuration of layers in the Neural Network
NN_layers = []#[200,200,100,100,50,50]
# Number of epochs, or total cycles over all batches
NN_epochs = 10
# Learning Rate
learning_rate = 0.01
# Ridge Regularization Parameter
regularization_param = 0
# File in which to save the terminal output
terminal_output_file = "log.txt"
# Directory in which to save the terminal output; underscore allows for
# automatic numbering
dirname = "results_cc_"
# If we want to load an older model, we can pass a string to the following
loadname = None
# Random Seed
rand_seed = 112358

""" IMPORTING COMMAND-LINE ARGUMENT SETTINGS """

# Initializing command-line arguments; NOTE all keys must be lowercase!
all_args = {"save":[str, "dirname"], "load":[str, "loadname"],
            "display":[int, "N_display"], "epochs":[int, "NN_epochs"],
            "batchsize":[int, "batchsize"], "lr":[float,"learning_rate"],
            "reg":[float,"regularization_param"], "seed":[int, "rand_seed"]}
parse_args(all_args)

np.random.seed(rand_seed)

""" DATASET PARAMETERS """

# Determines which columns in the dataset are categorical, and which values
# the categories take
categorical_cols = {"1":[1,2], "2":[1,2,3,4], "3":[1,2,3]}

""" READING THE DATA (READING FROM .xls FILE) """

t0 = time()
filename = "ccdata.xls"
print(f"\nReading data from file {filename}")
df = pd.read_excel(filename)
X_labels = list(df.iloc()[0,:-1])
Y_label = list(df.iloc()[0,-1])
X = np.array(df.iloc()[1:,1:-1], dtype = np.int64)
Y = np.array(df.iloc()[1:,-1], dtype = np.int64)[:,np.newaxis]
del df

""" PREPROCESSING, SPLITTING, AND RESHAPING THE DATA """

# Splitting the data into training and testing sets
X_train, Y_train, X_test, Y_test = split(X, Y, test_percent)
# Implements normalization and one-hot encoding
X_train, Y_train = preprocess(X_train, Y_train, categorical_cols)
X_test, Y_test = preprocess(X_test, Y_test, categorical_cols)
# Upsamples the training data
X_train, Y_train = upsample_binary(X_train, Y_train)

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
        f"Random Seed: {rand_seed:d}\n")
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
    reg = regularization_param, batchsize = batchsize)
else:
    print(f"Loading from directory <{loadname}>\n")
    # Loading a previous neural network
    NN.load(loadname)

# Predicting outputs for the testing data
Y_predict = NN.predict(X_test)

""" ERROR ANALYSIS """

# Rounding the predicted values to zero and one for sigmoid
Y_predict[Y_predict >= 0.5] = 1
Y_predict[Y_predict < 1] = 0

# Calculating the elementwise difference in the predicted and expected outputs
diffs = Y_predict-Y_test

# Displaying the total correct and incorrect outputs
incorrect = np.sum(np.abs(diffs))
correct = diffs.shape[0] - incorrect
total = diffs.shape[0]
msg2 = (f"\nTest Results\n\n\tNumber of incorrect outputs: {incorrect:.0f}/"
       f"{total:d}\n\tNumber of correct outputs: {correct:.0f}/{total:d}\n\t"
       f"Accuracy Score: {correct/total:.2g}\n\n"
       f"Prediction information:\n\n\tNo. of ones:\t{np.sum(Y_predict):.0f}"
       f"\n\tNo. of zeros:\t{Y_predict.shape[0] - np.sum(Y_predict):.0f}\n")
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
        outfile.write(msg1 + msg2)
