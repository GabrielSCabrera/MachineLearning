from neuralnet import NeuralNet
import pandas as pd
import numpy as np
from time import time

def preprocess(X, categorical_cols, delete_outliers = True):
    """
        categorical_cols should be a dict with:
        {"index_1"              :       [cat_11, cat_12, cat_13, ...],
         "index_2 index_3"      :       [cat_21, cat_22, cat_23, ...]}

        Where index_1, index_2, ... must always be of type <int>,
        Each index_i must be unique!

        One-hot encoding sorts the input cat_ij from least to greatest, and
        assigns new column indices 0,1,... to each cat_ij based on this order.
    """
    N,M = X.shape
    X_new = []
    categories = {}
    del_rows = []
    for i,j in categorical_cols.items():
        cat_keys = i.split(" ")
        cat_vals = np.sort(j)
        for k in cat_keys:
            categories[k] = cat_vals
    for i in range(M):
        if str(i) in categories.keys():
            key = str(i)
            val = categories[key]
            valmap = {v:n for n,v in enumerate(val)}
            new_cols = np.zeros((len(val), N))
            col = X[:,i]
            for n,c in enumerate(col):
                if int(c) not in val:
                    if delete_outliers is True:
                        del_rows.append(n)
                    else:
                        msg = (f"Found outlier {int(c)} at index ({n}, {k})\n"
                               f"Expected values: {val}")
                        raise Exception(msg)
                else:
                    new_cols[valmap[c],n] = 1
            for j in new_cols:
                X_new.append(j)
        else:
            col = X[:,i]
            shift = np.min(col)
            scale = np.max(col) - shift
            X_new.append((col - shift)/scale)
    del_rows = np.sort(del_rows)
    for row in del_rows[::-1]:
        X_new = np.delete(X_new, row, axis = 1)
    return np.array(X_new).T

t0 = time()

# Reading the Excel Data
filename = "ccdata.xls"
df = pd.read_excel(filename)
X_labels = list(df.iloc()[0,:-1])
Y_label = list(df.iloc()[0,-1])
categorical_cols = {"1":[1,2], "2":[1,2,3,4], "3":[1,2,3]}
X = np.array(df.iloc()[1:,1:-1], dtype = np.int64)
Y = np.array(df.iloc()[1:,-1], dtype = np.int64)
X = preprocess(X, categorical_cols)
del df

test_size = 3
trainlen = len(X)-test_size
trainlen = int(trainlen)
testlen = len(X)-trainlen

NN = NeuralNet(X[:trainlen], Y[:trainlen])
out = NN.learn(cycles = 1000, layers = [16,14,12,8], batchsize = test_size)
out = NN.predict(X[trainlen:])
out[out >= 0.5] = 1
out[out < 1] = 0
diffs = out.flatten().astype(np.int64)-Y[trainlen:]
print(f"Number of wrong outputs: {np.sum(np.abs(diffs))}")
print(f"Number of correct outputs: {diffs.shape[0] - np.sum(np.abs(diffs))}")

print(f"Time Elapsed: {time() - t0:.0f}s")
