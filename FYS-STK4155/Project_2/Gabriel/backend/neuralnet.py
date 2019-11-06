import matplotlib.pyplot as plt
from time import time
import numpy as np
try:
    import cupy as cp
except ImportError:
    print("Warning: CUDA not supported on this device")
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

def preprocess(X, Y, categorical_cols, delete_outliers = True, output_activation_fxn = "sigmoid"):
    """
        categorical_cols should be a dict with:
        {"index_1"              :       [cat_11, cat_12, cat_13, ...],
         "index_2 index_3"      :       [cat_21, cat_22, cat_23, ...]}

        Where index_1, index_2, ... must always be of type <int>,
        Each index_i must be unique!

        One-hot encoding sorts the input cat_ij from least to greatest, and
        assigns new column indices 0,1,... to each cat_ij based on this order.
    """
    if output_activation_fxn == "sigmoid":
        false_val = 0
        true_val = 1
    elif output_activation_fxn == "tanh":
        false_val = -1
        true_val = 1
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
            new_cols = np.ones((len(val), N))*false_val
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
                    new_cols[valmap[c],n] = true_val
            for j in new_cols:
                X_new.append(j)
        else:
            col = X[:,i]
            # Scaling based on mean and std
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

class NeuralNet:

    def __init__(self):
        self.attribute_reset()

    def attribute_reset(self):
        # Cleans out all attributes in case of reset
        if not hasattr(self, '_dir_backup'):
            self._dir_backup = self.__dir__().copy
        else:
            for var in self.__dir__():
                if var not in self._dir_backup() and var != "_dir_backup":
                    delattr(self, var)

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
        self.attribute_reset()

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

    def activation(self, function, x, GPU = False):
        """
            Parameter "function" can take values:
                function = "sigmoid"
                function = "tanh"
                function = "x"
        """
        if GPU is True:
            if function == "sigmoid":
                return 1/(1 + cp.exp(-x))
            elif function == "tanh":
                return cp.tanh(x)
            elif function == "x":
                return x
        else:
            if function == "sigmoid":
                return 1/(1 + np.exp(-x))
            elif function == "tanh":
                return np.tanh(x)
            elif function == "x":
                return x

    def diff_activation(self, function, x, GPU = False):
        """
            Parameter "function" can take values:
                function = "sigmoid"
                function = "tanh"
                function = "x"
            Assumes that the activation function outputs are passed in as "x",
            not the hidden layers pre-activation.
        """
        if GPU is True:
            if function == "sigmoid":
                return x*(1 - x)
            elif function == "tanh":
                return 1 - x**2
            elif function == "x":
                return cp.ones_like(x)
        else:
            if function == "sigmoid":
                return x*(1 - x)
            elif function == "tanh":
                return 1 - x**2
            elif function == "x":
                return np.ones_like(x)

    def train(self, epochs, layers, batchsize, lr = 0.01, reg = 0,
              activation_fxn = "sigmoid", output_activation_fxn = None,
              dtype = np.float64, GPU = False):
        """
        epochs:     Number of fwdfeed-backprop epochs <int>, minimum of 1
        layers:     List of layer sizes, each element (e >= 1) must be an <int>

        Note:       The bias term is added automatically.
        """

        if output_activation_fxn is None:
            output_activation_fxn = activation_fxn

        try:
            X, Y = self._X, self._Y
        except AttributeError:
            msg = "Must run method NeuralNet.set(), before NeuralNet.train()"
            raise AttributeError(msg)

        N, M = X.shape[0], Y.shape[0]
        P, Q = X.shape[1], Y.shape[1]

        epochs = int(epochs)
        layers = np.array(layers, dtype = np.int32)
        layers = np.concatenate([[P], layers, [Q]])

        rounded_len = (N//batchsize)*batchsize
        batches = N//batchsize

        X = X[:rounded_len,:,np.newaxis]
        Y = Y[:rounded_len,:,np.newaxis]

        X_batches = np.split(X, batches)
        Y_batches = np.split(Y, batches)

        if GPU is True:
            X_batches = cp.array(X_batches, dtype = dtype)
            Y_batches = cp.array(Y_batches, dtype = dtype)
        else:
            X_batches = np.array(X_batches)
            Y_batches = np.array(Y_batches)

        N = X_batches.shape[1]

        W = []
        B = []

        for i in range(len(layers)-1):
            W_step = np.random.normal(0, 0.5, (1, layers[i+1], layers[i]))
            B_step = np.random.normal(0, 0.5, (1, layers[i+1], 1))
            if GPU is True:
                W_step = cp.array(W_step, dtype = dtype)
                B_step = cp.array(B_step, dtype = dtype)
            W.append(W_step)
            B.append(B_step)

        perc = 0
        tot_iter = (epochs*len(X_batches))
        times = np.zeros(tot_iter)
        counter = 0
        t0 = time()
        dt = 0
        print(f"\t{0:>3d}%", end = "")

        if GPU is True:
            X = cp.array(X)
            Y = cp.array(Y)
            for e in range(epochs):
                for n in range(len(X_batches)):
                    X = X_batches[n]
                    Y = Y_batches[n]
                    Z = []
                    Z.append(X)
                    for i in layers[1:]:
                        Z.append(cp.zeros((N, i, 1)))

                    for m in range(len(W)):
                        w = W[m]
                        b = B[m]
                        Z[m+1] = w @ Z[m] + b
                        if m < len(W) - 1:
                            Z[m+1] = self.activation(activation_fxn, Z[m+1], GPU = GPU)
                        elif m == len(W) - 1:
                            Z[m+1] = self.activation(output_activation_fxn, Z[m+1], GPU = GPU)

                    delta = 2*(Z[-1] - Y)
                    delta *= self.diff_activation(output_activation_fxn, Z[-1], GPU = GPU)
                    for i in range(1, len(Z)):
                        dW = cp.einsum("ijk,ikj->ijk", delta, Z[-i-1])
                        W[-i] -= lr*(cp.mean(dW, axis = 0) - reg*W[-i])
                        B[-i] -= lr*cp.mean(delta, axis = 0)
                        W_T = cp.reshape(W[-i], (W[-i].shape[0], W[-i].shape[2], W[-i].shape[1]))
                        delta = W_T @ delta
                        delta *= self.diff_activation(activation_fxn, Z[-i-1], GPU = GPU)

                    counter += 1
                    new_perc = int(100*counter/tot_iter)
                    times[counter-1] = time()
                    if int(time() - t0) > dt:
                        perc = new_perc
                        t_avg = np.mean(np.diff(times[:counter]))
                        eta = t_avg*(tot_iter - counter)
                        hh = int(eta//3600)
                        mm = int((eta//60)%60)
                        ss = int(eta%60)
                        msg = f"\r\t{perc:>3d}% – ETA {hh:02d}:{mm:02d}:{ss:02d}"
                        print(msg, end = "")
                    dt = time() - t0
            for n in range(len(W)):
                W[n] = cp.asnumpy(W[n])
                B[n] = cp.asnumpy(B[n])
        else:
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
                        if m < len(W) - 1:
                            Z[m+1] = self.activation(activation_fxn, Z[m+1], GPU = GPU)
                        elif m == len(W) - 1:
                            Z[m+1] = self.activation(output_activation_fxn, Z[m+1], GPU = GPU)

                    delta = 2*(Z[-1] - Y)
                    delta *= self.diff_activation(output_activation_fxn, Z[-1], GPU = GPU)
                    for i in range(1, len(Z)):
                        dW = np.einsum("ijk,ikj->ijk", delta, Z[-i-1])
                        W[-i] -= lr*(np.mean(dW, axis = 0) - reg*W[-i])
                        B[-i] -= lr*np.mean(delta, axis = 0)
                        W_T = W[-i].reshape(1, W[-i].shape[2], W[-i].shape[1])
                        delta = W_T @ delta
                        delta *= self.diff_activation(activation_fxn, Z[-i-1], GPU = GPU)

                    counter += 1
                    new_perc = int(100*counter/tot_iter)
                    times[counter-1] = time()
                    if int(time() - t0) > dt:
                        perc = new_perc
                        t_avg = np.mean(np.diff(times[:counter]))
                        eta = t_avg*(tot_iter - counter)
                        hh = int(eta//3600)
                        mm = int((eta//60)%60)
                        ss = int(eta%60)
                        msg = f"\r\t{perc:>3d}% – ETA {hh:02d}:{mm:02d}:{ss:02d}"
                        print(msg, end = "")
                    dt = time() - t0
        dt = time() - t0
        hh = int(dt//3600)
        mm = int((dt//60)%60)
        ss = int(dt%60)
        print(f"\r\t100% – Total Time Elapsed {hh:02d}:{mm:02d}:{ss:02d}")

        self.W = W
        self.B = B

        return self.W, self.B

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

    def ROC(self, X_test, Y_test, savepath = None):

        if X_test.ndim == 1:
            X_test = X_test[:,np.newaxis].astype(np.float64)
        else:
            X_test = X_test.astype(np.float64)
        if Y_test.ndim == 1:
            Y_test = Y_test[:,np.newaxis].astype(np.float64)
        else:
            Y_test = Y_test.astype(np.float64)

        tot_bin = np.count_nonzero(Y_test == 0) + np.count_nonzero(Y_test == 1)
        binary_output = tot_bin == Y_test.size

        if binary_output is False:
            print("WARNING: May not work correctly for continuous outputs!!")

        N,p = X_test.shape
        M,q = Y_test.shape
        Y_predict = self.predict(X_test)
        for feature in range(q):
            Yf_test = Y_test[:,feature].flatten()
            Yf_predict = Y_predict[:,feature].flatten()
            Y_sort = Yf_predict.copy()
            Y_sort[Y_sort < 0.5] = Y_sort[Y_sort < 0.5] + 0.5
            idx = np.argsort(Y_sort)[::-1]

            Yf_predict[Yf_predict > 0.5] = 1
            Yf_predict[Yf_predict <= 0.5] = 0

            Yf_test = Yf_test[idx]
            Yf_predict = Yf_predict[idx]

            TP_arr = np.logical_and(np.equal(Yf_predict, Yf_test),
                                    np.equal(Yf_test, 1))
            TN_arr = np.logical_and(np.equal(Yf_predict, Yf_test),
                                    np.equal(Yf_test, 0))
            FP_arr = np.logical_and(np.equal(Yf_predict, np.logical_not(Yf_test)),
                                    np.equal(Yf_test, 0))
            FN_arr = np.logical_and(np.equal(Yf_predict, np.logical_not(Yf_test)),
                                    np.equal(Yf_test, 1))

            N = np.arange(1, len(Yf_test)+1, dtype = np.uint32)
            tpr = np.zeros_like(Yf_test, dtype = np.float64)
            fpr = np.zeros_like(Yf_test, dtype = np.float64)

            for i,(TP, TN, FP, FN) in enumerate(zip(TP_arr, TN_arr, FP_arr, FN_arr)):
                if TP + FN != 0:
                    tpr[i:] = tpr[i] + TP/(TP + FN)
                if FP + TN != 0:
                    fpr[i:] = fpr[i] + FP/(FP + TN)

            # Normalization constant
            norm = 1

            tpr = norm*(tpr - np.min(tpr))/(np.max(tpr) - np.min(tpr))
            fpr = norm*(fpr - np.min(fpr))/(np.max(fpr) - np.min(fpr))

            # Integral of Model
            AUC = np.trapz(tpr, fpr)
            # Baseline
            xBl = [np.min(fpr), np.max(fpr)]
            yBl = [np.min(tpr), np.max(tpr)]

            # Model
            if q == 1:
                model_label = "Model"
                AUC_text = f"AUC = {AUC:.2f}"
                x_text = norm*0.7
                y_text = norm*0.1
                plt.text(x_text, y_text, AUC_text, fontsize = 14)
                plt.plot(fpr, tpr, "k-", label = model_label)
            else:
                model_label = f"Model Feature {feature+1}; AUC = {AUC:.2f}"
                plt.plot(fpr, tpr, label = model_label)

        plt.plot(xBl, yBl, "b-.", label = "Baseline")
        plt.legend()
        plt.axis([0,1,0,1])
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.grid()
        if savepath is None:
            plt.show()
        else:
            plt.savefig(savepath, dpi = 250)
            plt.close()
