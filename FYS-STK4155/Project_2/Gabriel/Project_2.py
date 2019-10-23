from neuralnet import NeuralNet, preprocess
from time import time
import pandas as pd
import numpy as np
import os

""" PROGRAM PARAMETERS """

# Size of each batch sent into the neural network
batchsize = 50
# Percentage of data to set aside for testing
test_percent = 25
# Configuration of layers in the Neural Network
NN_layers = [200,100,75,50,25,10]
# Number of epochs, or total cycles over all batches
NN_epochs = 1
# File in which to save the terminal output
terminal_output_file = "term_out.txt"

""" DATASET PARAMETERS """

# Determines which columns in the dataset are categorical, and which values
# the categories take
categorical_cols = {"1":[1,2], "2":[1,2,3,4], "3":[1,2,3]}

""" READING THE DATA (READING FROM .xls FILE) """

t0 = time()

filename = "ccdata.xls"
df = pd.read_excel(filename)
X_labels = list(df.iloc()[0,:-1])
Y_label = list(df.iloc()[0,-1])
X = np.array(df.iloc()[1:,1:-1], dtype = np.int64)
Y = np.array(df.iloc()[1:,-1], dtype = np.int64)[:,np.newaxis]
del df

""" PREPROCESSING, SPLITTING, AND RESHAPING THE DATA """

# Implements normalization and one-hot encoding
X,Y = preprocess(X,Y, categorical_cols)

# Splitting the data into training and testing sets
N = X.shape[0]
N_train = N - (N*test_percent)//100
X_train, Y_train = X[:N_train], Y[:N_train]
X_test, Y_test = X[N_train:], Y[N_train:]

""" INFORMATION FOR THE USER """

msg1 = (f"\nProcessed Dataset Dimensions:\n"
        f"\n\tX_train: (N= {X_train.shape[0]}, p= {X_train.shape[1]})"
        f"\n\tY_train: (M= {Y_train.shape[0]}, q= {Y_train.shape[1]})\n"
        f"\n\tX_test:  (N= {X_test.shape[0]}, p= {X_test.shape[1]})"
        f"\n\tY_test:  (M= {Y_test.shape[0]}, q= {Y_test.shape[1]})\n"
        f"\nNeural Network Parameters\n"
        f"\n\tBatch Size: {batchsize}\n\tLayer Configuration: {NN_layers}"
        f"\n\tEpochs: {NN_epochs}\n\nTraining the Network:\n")
print(msg1)

""" IMPLEMENTING THE NEURAL NETWORK """

# Initializing the neural network with the training data
NN = NeuralNet(X_train, Y_train)

# Training the neural network with the parameters given earlier
W,B = NN.train(epochs = NN_epochs, layers = NN_layers, batchsize = batchsize)

# Predicting outputs for the testing data
Y_predict = NN.predict(X_test)

""" ERROR ANALYSIS """

# Rounding the predicted values to zero and one
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
       f"Percent correct: {100*correct/total:.0f}%")
print(msg2)

""" SAVNG DATA """

dirname = "results_"
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

for layer in range(len(NN_layers)):
    np.save(f"{dirname}/W/{filename}W_{layer:03.0f}", W[layer])
    np.save(f"{dirname}/B/{filename}B_{layer:03.0f}", B[layer])

with open(f"{dirname}/{terminal_output_file}", "w+") as outfile:
    outfile.write(msg1 + "\n" + msg2 + "\n")
