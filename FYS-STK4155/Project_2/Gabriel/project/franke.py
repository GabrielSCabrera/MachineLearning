from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from time import time
import numpy as np
import sys
import os

sys.path.append("..")
from backend.neuralnet import NeuralNet, split

t0 = time()

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def generate_Franke_data(x_min = 0, x_max = 1, N = 250):

    # Generating NxN meshgrid of x,y values in range [x_min, x_max]
    X = np.random.random((N,N))*(x_max-x_min) + x_min
    Y = np.random.random((N,N))*(x_max-x_min) + x_min

    # Calculating the values of the Franke function at each (x,y) coordinate
    Z = FrankeFunction(X,Y)
    init_error = np.random.normal(0, globals()["sigma"], Z.shape)
    f_xy = Z.copy().flatten()
    Z = Z + init_error

    # Making compatible input arrays for Regression object
    x = np.zeros((X.size, 2))
    x[:,0] = X.flatten()
    x[:,1] = Y.flatten()
    y = Z.reshape((Z.size, 1))
    # y = Z.flatten()[:,np.newaxis]

    return x, y, f_xy

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

""" PROGRAM PARAMETERS """

# Size of each batch sent into the neural network
batchsize = 100
# Percentage of data to set aside for testing
test_percent = 25
# Configuration of layers in the Neural Network
NN_layers = [100,100]#[575,383,255,170,113,75,50]
# Number of epochs, or total cycles over all batches
NN_epochs = 100
# Learning rate
learning_rate = 0.01
# Ridge regularization parameter
regularization_param = 1E-7
# Activation function
activation_fxn = "tanh"
# Activation function for output layer (None defaults to "activation_fxn")
output_activation_fxn = "x"
# File in which to save the terminal output
terminal_output_file = "log.txt"
# Optimize for CUDA
GPU = False
# Directory in which to save the terminal output; underscore allows for
# automatic numbering
dirname = "results_franke_"
# If we want to load an older model, we can pass a string to the following
loadname = None
# Gaussian noise in Franke function
sigma = 0.1
# Random Seed
rand_seed = 112358
# Plot filename
franke_img = None

""" IMPORTING COMMAND-LINE ARGUMENT SETTINGS """

# Initializing command-line arguments; NOTE all keys must be lowercase!
all_args = {"save":[str, "dirname"], "load":[str, "loadname"],
            "display":[int, "N_display"], "epochs":[int, "NN_epochs"],
            "batchsize":[int, "batchsize"], "lr":[float,"learning_rate"],
            "reg":[float, "regularization_param"], "seed":[int, "rand_seed"],
            "activation":[str, "activation_fxn"],
            "activation_out":[str, "output_activation_fxn"],
            "GPU":[bool, "GPU"], "saveimg":[str, "franke_img"]}
parse_args(all_args)

np.random.seed(rand_seed)

""" GENERATING, PREPROCESSING, SPLITTING, AND RESHAPING THE DATA """

if output_activation_fxn is None:
    output_activation_fxn = activation_fxn

X, Y, f_xy = generate_Franke_data(N = 150)
# Splitting the data into training and testing sets
X_train, Y_train, X_test, Y_test = split(X, Y, test_percent)


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
    # Inserting the training data into the network
    NN.set(X_train, Y_train)

    # Implements normalization and one-hot encoding
    NN.preprocess({}, False, output_activation_fxn)

    # Upsamples the training data
    NN.upsample_binary()

    print("Training the Network:\n")

    # Training the neural network with the parameters given earlier
    W,B = NN.train(epochs = NN_epochs, layers = NN_layers, lr = learning_rate,
    reg = regularization_param, batchsize = batchsize, GPU = GPU,
    activation_fxn = activation_fxn,
    output_activation_fxn = output_activation_fxn)

    scale_shift = NN._scale_shift
    scale_shift_X = np.array(scale_shift[:2])
    scale_shift_Y = np.array(scale_shift[2:])

else:
    print(f"Loading from directory <{loadname}>\n")
    # Loading a previous neural network
    NN.load(loadname)

# Predicting outputs for the testing data
Y_predict, Y_test = NN.predict(X_test, Y_test)

""" ERROR ANALYSIS """

# Calculating the elementwise difference in the predicted and expected outputs
MSE = np.mean((Y_predict-Y_test)**2)
SS_tot = np.sum((Y_test-np.mean(Y_test))**2)
SS_res = np.sum((Y_predict-Y_test)**2)
R2 = 1 - SS_res/SS_tot

# Displaying the total correct and incorrect outputs
msg2 = (f"\nTest Results\n\n\tMSE: {MSE:.4g}\n\tR²: {R2:.4g}")
print(msg2)

""" Plotting the test output """

""" SAVING DATA IF A NEW NETWORK IS CREATED"""

# if loadname is None:
#     if dirname[-1] == "_":
#         ID = 0.0
#         while True:
#             ID_text = f"{ID:03.0f}"
#             name_W = "W"
#             name_B = "B"
#             if os.path.isdir(dirname + ID_text):
#                 ID += 1
#             else:
#                 dirname = dirname + ID_text
#                 os.mkdir(dirname)
#                 os.mkdir(dirname + "/W")
#                 os.mkdir(dirname + "/B")
#                 break
#     elif not os.path.isdir(dirname):
#         os.mkdir(dirname)
#         os.mkdir(dirname + "/W")
#         os.mkdir(dirname + "/B")
#     else:
#         if not os.path.isdir(dirname + "/W"):
#             os.mkdir(dirname + "/W")
#         if not os.path.isdir(dirname + "/B"):
#             os.mkdir(dirname + "/B")
#
#     for layer in range(len(W)):
#         np.save(f"{dirname}/W/layer_{layer:03.0f}", W[layer])
#         np.save(f"{dirname}/B/layer_{layer:03.0f}", B[layer])
#
#     NN_layers = np.concatenate([[X_train.shape[1]], NN_layers, [Y_train.shape[1]]])
#     np.save(f"{dirname}/layers", NN_layers)
#
#     with open(f"{dirname}/{terminal_output_file}", "w+") as outfile:
#         outfile.write(msg1 + msg2)
#

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
    else:
        if not os.path.isdir(dirname + "/W"):
            os.mkdir(dirname + "/W")
        if not os.path.isdir(dirname + "/B"):
            os.mkdir(dirname + "/B")

    for layer in range(len(W)):
        np.save(f"{dirname}/W/layer_{layer:03.0f}", W[layer])
        np.save(f"{dirname}/B/layer_{layer:03.0f}", B[layer])

    NN_layers = np.concatenate([[X_train.shape[1]], NN_layers, [Y_train.shape[1]]])
    np.save(f"{dirname}/layers", NN_layers)
    np.save(f"{dirname}/scale_shift_X", scale_shift_X)
    np.save(f"{dirname}/scale_shift_Y", scale_shift_Y)

    with open(f"{dirname}/{terminal_output_file}", "w+") as outfile:
        outfile.write(msg1 + msg2)

    with open(f"{dirname}/activations.dat", "w+") as outfile:
        outfile.write(f"{activation_fxn} {output_activation_fxn}")

    with open(f"{dirname}/categorical_cols.dat", "w+") as outfile:
        string = ""
        for key,val in {}.items():
            rhs = ""
            for i in val:
                rhs += f"{i} "
            string += f"{key}:{rhs[:-1]}\n"
        outfile.write(string[:-1])

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    fig.set_size_inches(8, 6)
    fig.tight_layout()
    ax.plot(X_test[:,0], X_test[:,1], Y_test[:,0], "b.", label = "Expected")
    ax.plot(X_test[:,0], X_test[:,1], Y_predict[:,0], "r.", label = "Predicted")
    plt.legend()
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$f(x,y)$")
    if franke_img is None:
        plt.show()
    else:
        plt.savefig(f"{dirname}/{franke_img}", dpi = 250)
        plt.close()
