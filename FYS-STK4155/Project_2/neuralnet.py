from multiprocessing import Pool
import matplotlib.pyplot as plt
from imageio import imread
from numba import jit
import numpy as np

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

    def learn(self, cycles, nodes, layers, lr = 0.1, mu = 0, sigma = 1):
        """
        cycles: Number of for-back cycles <int>, minimum of 1
        nodes: Number of nodes <int>, minimum of 1
        layers: Number of hidden layers <int>, minimum of 1

        Note: The bias term is added automatically.
        """
        X, Y = self._X.flatten()[:,np.newaxis], self._Y.flatten()[:,np.newaxis]

        scale = 1
        shift = 0

        cycles = int(cycles)
        nodes = int(nodes)
        layers = int(layers)

        if np.min(Y) != 0:
            shift = -np.min(Y)

        if np.max(Y) - np.min(Y) != 1:
            scale = np.max(Y)-np.min(Y)

        Y = (Y + shift)/scale

        N, M = X.shape[0], Y.shape[0]

        W_in = np.random.normal(mu, sigma, (nodes, N))
        W = np.random.normal(mu, sigma, (layers, nodes, nodes))
        W_out = np.random.normal(mu, sigma, (M, nodes))

        B_in = np.random.normal(mu, sigma, (nodes, 1))
        B = np.random.normal(mu, sigma, (layers, nodes, 1))
        B_out = np.random.normal(mu, sigma, (M, 1))

        Z = np.zeros((layers, nodes, 1), dtype = np.float64)

        @jit(cache = True, nopython = True)
        def feedforward(X, Z, W_in, W, W_out, B_in, B, B_out, layers):
            Z[0] = W_in @ X + B_in
            Z[0] = 1/(1 + np.exp(-Z[0]))
            for l in range(layers-1):
                Z[l+1] = W[l] @ Z[l] + B[l]
                Z[l+1] = 1/(1 + np.exp(-Z[l+1]))
            A = W_out @ Z[-1] + B_out
            A = 1/(1 + np.exp(A))
            return A, Z

        @jit(cache = True, nopython = True)
        def backprop(X, Y, A, Z, W_in, W, W_out, B_in, B, B_out, layers, lr):
            diff = A - Y
            C = np.mean(diff**2)

            dCdA = 2*(A - Y)
            dAdZ = A*(A - 1)
            dZdW = Z[-1]
            delta = dCdA*dAdZ
            W_out -= lr*delta @ dZdW.T
            B_out -= lr*delta

            delta = (W_out.T @ delta)*(Z[-1]*(1 - Z[-1]))
            W[-1] -= lr*delta @ Z[-1].T
            B[-1] -= lr*delta

            for i in range(1, layers):
                l = layers - i - 1
                delta = (W[l+1].T @ delta)*(Z[l]*(1 - Z[l]))
                W[l] -= lr*delta @ Z[l].T
                B[l] -= lr*delta

            delta = W[0].T @ delta
            W_in -= lr*delta @ X.T
            B_in -= lr*delta

            return W_in, W, W_out, B_in, B, B_out

        perc = 0
        print(f"\r{perc:>3d}%", end = "")
        for c in range(cycles):
            A, Z = feedforward(X, Z, W_in, W, W_out, B_in, B, B_out, layers)
            if c == cycles-1:
                break
            W_in, W, W_out, B_in, B, B_out = \
            backprop(X, Y, A, Z, W_in, W, W_out, B_in, B, B_out, layers, lr)
            new_perc = int(100*(c+1)/cycles)
            if new_perc > perc:
                perc = new_perc
                print(f"\r{perc:>3d}%", end = "")
        print(f"\r{100:>3d}%")

        self.W_in = W_in
        self.W = W
        self.W_out = W_out

        self.B_in = B_in
        self.B = B
        self.B_out = B_out

        self.layers = layers
        self.nodes = nodes
        self.scale = scale
        self.shift = shift

        return (np.reshape(A, self._X.shape)*self.scale)+self.shift

    def predict(self, X):

        X = X.flatten()[:,np.newaxis].astype(np.float64)

        W_in, W, W_out, B_in, B, B_out =\
        self.W_in, self.W, self.W_out, self.B_in, self.B, self.B_out

        layers = self.layers
        nodes = self.nodes
        Z = np.zeros((layers, nodes, 1), dtype = np.float64)

        @jit(cache = True, nopython = True)
        def feedforward(X, Z, W_in, W, W_out, B_in, B, B_out, layers):
            Z[0] = W_in @ X + B_in
            Z[0] = 1/(1 + np.exp(Z[0]))
            for l in range(layers-1):
                Z[l+1] = W[l] @ Z[l] + B[l]
                Z[l+1] = 1/(1 + np.exp(Z[l+1]))
            A = W_out @ Z[-1] + B_out
            A = 1/(1 + np.exp(A))
            return A, Z

        A, Z = feedforward(X, Z, W_in, W, W_out, B_in, B, B_out, layers)
        return (A*self.scale)+self.shift

if __name__ == "__main__":

    print("Importing")
    X = np.array(imread("a.jpg"))
    Y = np.array(imread("b.jpg"))

    # X = np.mean(X, axis = 2)
    # Y = np.mean(Y, axis = 2)

    NN = NeuralNet(X, Y)

    print("Learning")
    Z = NN.learn(cycles = 1000, nodes = 5, layers = 2, lr = 1E-3, mu = 0, sigma = 0.001)

    print("Plotting")
    plt.imshow(Z.astype(np.int32))
    plt.savefig("output.png")
    plt.close()
