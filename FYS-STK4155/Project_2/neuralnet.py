from numba.typed import List as jitlist
import matplotlib.pyplot as plt
from numba import njit
import numpy as np
import sys

def shapes(*args):
    for i in args:
        print(i.shape, end = " ")
    print()

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

        cycles = int(cycles)
        layers = np.array(layers, dtype = np.int64)
        layers = np.concatenate([[P], layers, [Q]])

        rounded_len = (N//batchsize)*N
        batches = N//batchsize

        X_batches = np.split(X[:rounded_len-1], batches)
        Y_batches = np.split(Y[:rounded_len-1], batches)

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
                W.append(np.random.random((layers[i+1], layers[i])))
                B.append(np.random.random((layers[i+1], 1)))

        # @njit(cache = True)
        def wrapped(X_batches, Y_batches, W, B, lr, cycles, layers):
            perc = 0
            N = X_batches[0].shape[0]
            print("0%")
            for c in range(cycles):
                for n in range(len(X_batches)):
                    X = X_batches[n].T
                    Y = Y_batches[n].T
                    Z = []
                    # Z = jitlist()
                    Z.append(X)
                    for i in layers:
                        Z.append(np.zeros((N, i, 1)))

                    for m in range(len(W)):
                        w = W[m]
                        b = B[m]
                        Z[m+1] = w @ Z[m] + b
                        Z[m+1] = 1/(1 + np.exp(-Z[m+1]))

                    dCdA = -2*(Y - Z[-1])
                    shapes(dCdA, Y, Z[-1])
                    dAdZ = Z[-1]*(1 - Z[-1])
                    shapes(dAdZ, Z[-1])
                    delta = dCdA*dAdZ

                    if c == cycles-1:
                        break

                    for m in range(len(W)-1):
                        l = len(W)-m-1
                        shapes(W[l], delta, Z[l].T)
                        shapes(B[l], delta)
                        dW = lr*delta @ Z[l].T
                        dB = lr*delta
                        W[l] += np.mean(dW, axis = 0)
                        B[l] += np.mean(dB, axis = 0)
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
