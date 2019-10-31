import matplotlib.pyplot as plt
from time import time
import numpy as np
import sys, os

# np.einsum("ijk,ikj->ijk",a,b) # DO NOT LOSE THIS!

def split(X, Y, test_percent = 0.25):
    N = X.shape[0]
    N_train = N - (N*test_percent)//100

    train_idx = np.random.choice(X.shape[0], N_train)
    test_idx = np.delete(np.arange(0, X.shape[0]), train_idx)

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    return X_train, Y_train, X_test, Y_test

def preprocess(X, Y, categorical_cols, delete_outliers = True):
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
            # Scaling based on mean and std
            col = X[:,i]
            X_new.append((col - np.mean(col))/np.std(col))
    del_rows = np.sort(del_rows)
    for row in del_rows[::-1]:
        X_new = np.delete(X_new, row, axis = 1)
        Y = np.delete(Y, row, axis = 0)
    return np.array(X_new).T, Y

def upsample_binary(X, Y):
    """
        Creates "new" datapoints from previously existing data, so as to
        make the distribution of outputs Y half-zero and half-one.  Helps to
        prevent the network from being biased one way or the other.

        Warning: may drastically increase the array sizes
    """
    tol = 1E-15
    idx_zeros = np.argwhere(np.abs(Y - 0) < tol)[:,0]
    idx_ones = np.argwhere(np.abs(Y - 1) < tol)[:,0]
    N_zeros = len(idx_zeros)
    N_ones = len(idx_ones)
    if N_zeros > N_ones:
        ratio = int(N_zeros//N_ones)
        new_len = ratio*N_ones + N_zeros
        X_up = np.zeros((new_len, X.shape[1]))
        Y_up = np.zeros((new_len, Y.shape[1]))
        sample_idx = np.random.choice(idx_ones, (ratio - 1)*N_ones)
        X_up[:X.shape[0]] = X
        Y_up[:Y.shape[0]] = Y
        X_up[X.shape[0]:] = X[sample_idx]
        Y_up[Y.shape[0]:] = Y[sample_idx]
    elif N_zeros < N_ones:
        ratio = int(N_ones//N_zeros)
        new_len = ratio*N_zeros + N_ones
        X_up = np.zeros((new_len, X.shape[1]))
        Y_up = np.zeros((new_len, Y.shape[1]))
        sample_idx = np.random.choice(idx_zeros, (ratio - 1)*N_zeros)
        X_up[:X.shape[0]] = X
        Y_up[:Y.shape[0]] = Y
        X_up[X.shape[0]:] = X[sample_idx]
        Y_up[Y.shape[0]:] = Y[sample_idx]
    else:
        X_up, Y_up = X, Y

    shuffle_idx = np.random.choice(X_up.shape[0], (X_up.shape[0]))
    X_up = X_up[shuffle_idx]
    Y_up = Y_up[shuffle_idx]
    return X_up, Y_up

def shapes(*args):
    """
        For quick bugfixing
    """
    for i in args:
        print(i.shape, end = " ")
    print()

class NeuralNet:

    def __init__(self):
        pass

    def load(self, path):

        # Loading the layer configuration
        layers = np.load(f"{path}/layers.npy")

        # Loading the weights
        W_filenames = np.sort(os.listdir(f"{path}/W/"))

        msg = (f"Data corrupted – inconsistencies detected in saved data\n"
        f"\tdir: {path}/W")
        if len(W_filenames) != len(layers)-1:
            raise ValueError(msg)

        W = []
        for f,l,l_prev in zip(W_filenames, layers[1:], layers[:-1]):
            W.append(np.load(f"{path}/W/{f}"))
            if W[-1].shape[0] != 1 or W[-1].shape[1] != l or W[-1].shape[2] != l_prev:
                raise ValueError(msg)

        # Loading the biases
        B_filenames = np.sort(os.listdir(f"{path}/B/"))

        msg = (f"Data corrupted – inconsistencies detected in saved data\n"
        f"\tdir: {path}/B")
        if len(B_filenames) != len(layers)-1:
            raise ValueError(msg)

        B = []
        for f,l in zip(B_filenames, layers[1:]):
            B.append(np.load(f"{path}/B/{f}"))
            if B[-1].shape[0] != 1 or B[-1].shape[1] != l or B[-1].shape[2] != 1:
                raise ValueError(msg)

        self.W = W
        self.B = B

    def set(self, X, Y):
        """
            IN
            X:          NumPy Array (N, p) or (N,)
            Y:          NumPy Array (M, q) or (M,)

            OUT
            self._X:    NumPy Array (N, p)
            self._Y:    NumPy Array (M, q)
        """
        if X.ndim == 1:
            self._X = X[:,np.newaxis].astype(np.float64)
        else:
            self._X = X.astype(np.float64)
        if Y.ndim == 1:
            self._Y = Y[:,np.newaxis].astype(np.float64)
        else:
            self._Y = Y.astype(np.float64)
        self._N = self._X.shape[0]
        self._M = self._Y.shape[0]
        self._p = self._X.shape[1]
        self._q = self._Y.shape[1]

    def train(self, epochs, layers, batchsize, lr = 0.01):
        """
        epochs:     Number of fwdfeed-backprop epochs <int>, minimum of 1
        layers:     List of layer sizes, each element (e >= 1) must be an <int>

        Note:       The bias term is added automatically.
        """
        try:
            X, Y = self._X, self._Y
        except AttributeError:
            msg = "Must run method NeuralNet.set(), before NeuralNet.train()"
            raise AttributeError(msg)

        N, M = X.shape[0], Y.shape[0]
        P, Q = X.shape[1], Y.shape[1]

        epochs = int(epochs)
        layers = np.array(layers, dtype = np.int64)
        layers = np.concatenate([[P], layers, [Q]])

        rounded_len = (N//batchsize)*batchsize
        batches = N//batchsize

        X_batches = np.split(X[:rounded_len,:,np.newaxis], batches)
        Y_batches = np.split(Y[:rounded_len,:,np.newaxis], batches)

        Y_shift = 0
        Y_scale = 1

        if np.min(Y) != 0:
            Y_shift = -np.min(Y)

        if np.max(Y) - np.min(Y) != 1:
            Y_scale = np.max(Y)-np.min(Y)

        W = []
        B = []

        for i in range(len(layers)-1):
            W.append(np.random.normal(0, 0.5, (1, layers[i+1], layers[i])))
            B.append(np.random.normal(0, 0.5, (1, layers[i+1], 1)))

        perc = 0
        N = X_batches[0].shape[0]
        tot_iter = (epochs*len(X_batches))
        times = np.zeros(tot_iter)
        counter = 0
        t0 = time()
        dt = 0
        print(f"\t{0:>3d}%", end = "")
        for e in range(epochs):
            for n in range(len(X_batches)):
                X = X_batches[n]
                Y = Y_batches[n]
                Z = []
                Z.append(X)
                for i in layers[1:]:
                    Z.append(np.zeros((N, i, 1)))

                for m in range(len(W)):
                    w = W[m]
                    b = B[m]
                    Z[m+1] = w @ Z[m] + b
                    Z[m+1] = 1/(1 + np.exp(-Z[m+1]))        # sigmoid
                    # Z[m+1] = np.tanh(-Z[m+1])               # tanh

                dCdA = -2*(Y - Z[-1])
                dAdZ = Z[-1]*(1 - Z[-1])                    # sigmoid
                # dAdZ = 1 - np.tanh(Z[-1])**2                # tanh
                delta = dCdA*dAdZ

                for i in range(1, len(Z)):
                    dW = np.einsum("ijk,ikj->ijk", delta, Z[-i-1])
                    W[-i] -= lr*np.mean(dW, axis = 0)
                    B[-i] -= lr*np.mean(delta, axis = 0)
                    W_T = W[-i].reshape(1, W[-i].shape[2], W[-i].shape[1])
                    delta = W_T @ delta
                    delta *= Z[-i-1]*(1 - Z[-i-1])          # sigmoid
                    # delta *= 1 - np.tanh(Z[-i-1])**2        # tanh

                counter += 1
                new_perc = int(100*counter/tot_iter)
                times[counter-1] = time()
                if int(time() - t0) > dt:
                    perc = new_perc
                    t_avg = np.mean(np.diff(times[:counter]))
                    eta = t_avg*(tot_iter - counter)
                    hh = eta//3600
                    mm = (eta//60)%60
                    ss = eta%60
                    msg = f"\r\t{perc:>3d}% – ETA {hh:02.0f}:{mm:02.0f}:{ss:02.0f}"
                    print(msg, end = "")
                dt = time() - t0
        dt = time() - t0
        hh = dt//3600
        mm = (dt//60)%60
        ss = dt%60
        print(f"\r\t100% – Total Time Elapsed {hh:02.0f}:{mm:02.0f}:{ss:02.0f}")

        self.W = W
        self.B = B

        return W,B

    def predict(self, X):
        W = self.W
        B = self.B

        Z = X[:,:,np.newaxis]
        for m in range(len(W)):
            w = W[m]
            b = B[m]
            Z = w @ Z + b
            Z = 1/(1 + np.exp(-Z))                    # sigmoid
            # Z = np.tanh(Z)                              # tanh
        Z = np.squeeze(Z)
        if Z.ndim == 1:
            Z = Z[:,np.newaxis]
        return Z
