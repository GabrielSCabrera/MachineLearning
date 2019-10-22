from numba.typed import List as jitlist
import matplotlib.pyplot as plt
from numba import njit
import numpy as np
import sys

def shapes(*args):
    for i in args:
        print(i.shape, end = " ")
    print()

def preprocess(X):
    N,M = X.shape
    X_new = []
    for i in range(M):
        col = X[:,i]
        col_int = col.astype(int)
        if np.all((col - col_int) == 0) and np.max(col) < 10:
            categories = int(np.max(col) - np.min(col)) + 1
            col = col - np.min(col)
            new_cols = np.zeros((categories, N))
            for n,c in enumerate(col):
                new_cols[int(c),n] = 1
            for c in new_cols:
                X_new.append(c)
        else:
            shift = np.min(col)
            scale = np.max(col) - shift
            X_new.append((col - shift)/scale)
    return np.array(X_new).T

class NeuralNet:

    def __init__(self, X, Y):
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

        info = (f"X: (N= {self._N}, p= {self._p})"
                f"\nY: (M= {self._M}, q= {self._q})")
        print(info)

    def learn(self, cycles, layers, batchsize, lr = 0.1):
        """
        cycles:     Number of fwdfeed-backprop cycles <int>, minimum of 1
        layers:     List of layer sizes, each element (e >= 1) must be an <int>

        Note:       The bias term is added automatically.

        batchsize must be divisible into both the X and Y datasets
        """
        # X, Y = self._X.flatten()[:,np.newaxis], self._Y.flatten()[:,np.newaxis]
        X, Y = self._X, self._Y
        N, M = X.shape[0], Y.shape[0]
        P, Q = X.shape[1], Y.shape[1]

        if N >= M:
            A_split = M
            B_split = N
            txt_A = "Y"
            txt_B = "X"
        else:
            A_split = N
            B_split = M
            txt_A = "X"
            txt_B = "Y"
        if A_split%batchsize != 0:
            raise Exception(f"{txt_A} must be divisible into batchsize")
        splits = A_split//batchsize
        if B_split%splits != 0:
            raise Exception(f"{txt_B} must be divisible into batchsize")
        X_batches = np.array(np.split(X, splits))
        Y_batches = np.array(np.split(Y, splits))

        N_batch = N//splits
        M_batch = M//splits

        cycles = int(cycles)
        layers = np.array(layers, dtype = np.int64)
        layers = np.concatenate([[N_batch], layers, [M_batch]])

        Y_shift = 0
        Y_scale = 1

        if np.min(Y) != 0:
            Y_shift = -np.min(Y)

        if np.max(Y) - np.min(Y) != 1:
            Y_scale = np.max(Y)-np.min(Y)

        W = jitlist()
        B = jitlist()

        for i in range(len(layers)-1):
            if i < len(layers) - 1:
                W.append(np.random.random((P, layers[i+1], layers[i])))
                B.append(np.random.random((P, layers[i+1], 1)))

        # @njit(cache = True)
        def wrapped(X_batches, Y_batches, W, B, lr, cycles, layers):
            perc = 0
            print("0%")
            for c in range(cycles):
                for n in range(len(X_batches)):
                    X = X_batches[n]
                    Y = Y_batches[n]
                    Z = []
                    # Z = jitlist()
                    Z.append(X)
                    for i in layers:
                        Z.append(np.zeros((i, 1)))

                    for m in range(len(W)):
                        w = W[m]
                        b = B[m]
                        Z[m+1] = w @ Z[m] + b
                        Z[m+1] = 1/(1 + np.exp(-Z[m+1]))

                    dCdA = 2*(Y - Z[-1])
                    dAdZ = Z[-1]*(1 - Z[-1])
                    delta = dCdA*dAdZ

                    if c == cycles-1:
                        break

                    for m in range(len(W)-1):
                        l = len(W)-m-1
                        shapes(W[l], delta, Z[l].T)
                        W[l] += lr*delta @ Z[l].T
                        B[l] += lr*delta
                        delta = W[l].T @ delta
                        delta *= Z[l]*(1 - Z[l])

                    new_perc = int(100*(n+1 + c*len(X_batches))/(cycles*len(X_batches)))
                    if new_perc > perc:
                        perc = new_perc
                        print("\033[1A\r", perc, "\b%")
            print("\033[1A\r100%")
            return W,B

        W, B = wrapped(X_batches, Y_batches, W, B, lr, cycles, layers)

        self.W = W
        self.B = B

        self.layers = layers

        self.Y_scale = Y_scale
        self.Y_shift = Y_shift

        # return (np.reshape(A, self._Y.shape)*self.Y_scale)+self.Y_shift

    def predict(self, X):
        X = X.flatten()[:,np.newaxis]

        Y_shift = 0
        Y_scale = 1

        W = self.W
        B = self.B

        for n,(w,b) in enumerate(zip(W,B)):
            X = w @ X + b
            X = 1/(1 + np.exp(-X))              # Sigmoid

        return X
