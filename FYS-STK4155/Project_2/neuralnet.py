from multiprocessing import Pool
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

    def run(self, nodes, layers):
        """
        nodes: Number of nodes <int>, minimum of 1
        layers: Number of hidden layers <int>, minimum of 1

        Note: The bias term is added automatically.
        """
        mu, sigma = 0, 1
        X, Y = self._X, self._Y
        M, N = self._M, self._N
        p, q = self._p, self._q

        W_in = np.random.normal(mu, sigma, (nodes, N))
        W = np.random.normal(mu, sigma, (layers, nodes, nodes))
        W_out = np.random.normal(mu, sigma, (M, nodes))

        B_in = np.random.normal(mu, sigma, (nodes, p))
        B = np.random.normal(mu, sigma, (layers, nodes, p))
        B_out = np.random.normal(mu, sigma, (M, q))
        redim = np.random.normal(mu, sigma, (p, q))

        # Hidden Layers
        H = np.zeros((layers, nodes, p), dtype = np.float64)

        @jit(cache = True, nopython = True)
        def fast_forward(X, H, W_in, W, W_out, B_in, B, B_out, redim, layers):
            H[0] = W_in @ X + B_in
            H[0] = 1/(1 + np.exp(H[0]))
            for l in range(layers-1):
                H[l+1] = W[l] @ H[l] + B[l]
                H[l+1] = 1/(1 + np.exp(H[l+1]))
            Z = W_out @ H[l+1] + B_out
            Z = 1/(1 + np.exp(Z @ redim))
            return Z, H

        # @jit(cache = True, nopython = True)
        def backprop(Y, Z, H, W_in, W, W_out, B_in, B, B_out, redim, layers):
            Ey = Z - Y
            print(Ey.shape)
            MSE = np.mean(Ey**2)
            dZ = (1-Z)**2
            delta_Y = Ey*dZ     #ok
            dW_out = -redim.T*H[-1]
            print(dW_out.shape)


        Z, H = fast_forward(X, H, W_in, W, W_out, B_in, B, B_out, redim, layers)
        backprop(Y, Z, H, W_in, W, W_out, B_in, B, B_out, redim, layers)
if __name__ == "__main__":
    N = 100
    x = np.array([[0,0,0,0,1,0,0,1,0,0,1,1,1,0,0,1,1,1,1,1,0,1,1,1],
                  [0,0,1,0,0,0,0,0,0,0,1,1,0,1,1,1,1,1,1,1,1,0,1,1]])
    y = np.array( [0,0,1,0,1,0,0,1,0,0,0,0,1,1,0,0,1,0,0,0,1,1,0,0])
    NN = NeuralNet(x.T, y)
    X = NN.run(5,10)
    # print(X)
