from multiprocessing import Pool
import numpy as np

class NeuralNet:

    def __init__(self, X, Y):
        if X.ndim == 1:
            self._X = X[:,np.newaxis]
        else:
            self._X = X
        if Y.ndim == 1:
            self._Y = Y[:,np.newaxis]
        else:
            self._Y = Y
        self._N = self._X.shape[0]
        self._p = self._X.shape[1]
        self._q = self._Y.shape[1]

    def run(self, cycles, layers, alpha = 1E-3):
        w = 2*np.random.random((layers, self._N, self._N, 1)) - 1
        b = 2*np.random.random((layers, self._N)) - 1
        pool = Pool()
        net = np.zeros((layers, self._N, self._p))
        out = np.zeros((layers+1, self._N, self._p))
        out[0] = self._X
        for c in range(cycles):
            for i in range(layers):
                net[i], out[i+1] = self.layer(out[i], w[i], b[i], pool)

            delta = self.delta_output(out[-1], self._Y)
            print(out[-1].shape)
            for i in range(layers):
                idx = layers - i - 1
                for j in range(self._N):
                    delta = self.delta_inner(w[i], delta, out[i])
                    dw = delta*out[i]
                    print(w[i,j].shape, dw.shape)
                    w[i,j] -= alpha*dw

    def delta_output(self, out, target):
        dL = 2*(out - target)       # MSE
        dphi = out*(1-out)          # Sigmoid
        return np.mean(dL*dphi, axis = 1)

    def delta_inner(self, w, delta_prev, out):
        dL = np.sum(w*delta_prev)
        dphi = out*(1-out)
        return np.mean(dL*dphi, axis = 1)

    def MSE(self, out, target):
        return (out - target)**2

    def sigmoid(self, net):
        return 1/(1 + np.exp(-net))

    def layer(self, out_prev, w, b, pool):
        net = pool.starmap(self.perceptron,
        ([out_prev, w[i], b[i]] for i in range(out_prev.shape[0])))
        out = np.array(pool.map(self.sigmoid, net))
        return net, out

    def perceptron(self, X, w, b):
        return np.sum(w*X + b, axis = 0)

if __name__ == "__main__":
    N = 100
    x = np.array([[0,0,0,0,1,0,0,1,0,0,1,1,1,0,0,1,1,1,1,1,0,1,1,1],
                  [0,0,1,0,0,0,0,0,0,0,1,1,0,1,1,1,1,1,1,1,1,0,1,1]])
    y = np.array( [0,0,1,0,1,0,0,1,0,0,0,0,1,1,0,0,1,0,0,0,1,1,0,0])
    NN = NeuralNet(x.T, y)
    X = NN.run(2,10,50)
    # print(X)
