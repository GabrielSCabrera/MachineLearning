from neuralnet import NeuralNet
import pandas as pd
import numpy as np
from time import time

t0 = time()

# Reading the Excel Data
filename = "ccdata.xls"
df = pd.read_excel(filename)
X_labels = list(df.iloc()[0,:-1])
Y_label = list(df.iloc()[0,-1])
X = np.array(df.iloc()[1:,1:-1], dtype = np.int64)
Y = np.array(df.iloc()[1:,-1], dtype = np.int64)
del df

NN = NeuralNet(X, Y)
out = NN.learn(cycles = 1E3, nodes = 5, layers = 30)
out[out >= 0.5] = 1
out[out < 1] = 0
diffs = out.flatten().astype(np.int64)-Y
print(f"Number of wrong outputs: {np.sum(np.abs(diffs))}")
print(f"Number of correct outputs: {diffs.shape[0] - np.sum(np.abs(diffs))}")

print(f"Time Elapsed: {time() - t0:.0f}s")
