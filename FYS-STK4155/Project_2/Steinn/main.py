import lib.functions as fns
import lib.logistic_regression as lgr
import lib.neural_network as nnw
import sklearn
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
import dill
import seaborn as sb


def main():
    logistic_regression_study()
    # linear_regression_study()
    return None

def logistic_regression_study():
    # ------------------------- Data preparation -------------------------
    sd = int(time.time())
    fn = "defaulted_cc-clients.xls"
    # fns.read_in_data(fn)  # to preprocess and save features X and outcomes y
    Xf, yf = fns.load_features_predictors() # load preprocessed data

    # Xf = fns.onehotencode_data(Xf)
    # X, Xt, y, yt = sklearn.model_selection.train_test_split(
        # Xf, yf, test_size=0.2, random_state=sd, stratify=yf
    # )

    # X, means, stds = fns.scale_data(X, dtype='training')
    # Xt = fns.scale_data(Xt, mu=means, std=stds dtype='training')

    Xf = fns.PCA(Xf, dims_rescaled_data=21) # 21 from looking at the eigenvalues


    # ca. 77.88% of the data is zero. Requires resampling of training data
    Xf, yf = fns.upsample(Xf, yf, sd)
    # Xf, yf = fns.downsample(Xf, yf, sd)

    # Xf, yf = make_moons(n_samples=10000, noise=0.01)

    X, Xt, y, yt = sklearn.model_selection.train_test_split(
        Xf, yf, test_size=0.2, random_state=sd, stratify=yf
    )

    dfrac = -1  # portion of the data to analyse. must be between 1-30000
    X, Xt, y, yt = X[:dfrac], Xt[:dfrac], y[:dfrac], yt[:dfrac]

    # ----------------- Classification (credit card data) ----------------
    # Logistic Regression
    # print(SGD_with_minibatches(X, y, Xt, yt, gamma=0.1,
        # max_iter=1000, batch_size=10, verbose=False)) # our code
    # print(Sklearn_sgd_classifier(X, y, Xt, yt)) # comparison with sklearn
    # --------------------------------------------------------------------
    # Artificial Neural Networks
    # n1 = logistic_NNW(X, y, Xt, yt) # our code
    # plt.show()
    # Tensorflow_neural_network(X, y, Xt, yt) # comparison with tensorflow

    #   --------Train a Grid----------
    n1 = logistic_NNW(X, y, Xt, yt) # our code
    # n1.test_logistic_neuron(Xt,yt, load_data=False)
    self.X = Xt[:, :, np.newaxis]
    n1.output_func()

    gridsearch=0

    if gridsearch:
        Train_save_objects = 0
        Test_saved_objects = 1
        emin = 0; emax = -3; eres = 4  # resolution and limits of search
        hmin = 0; hmax = -7; hres = 8
        eta_array   = np.logspace(emin, emax, eres)
        hyperp_array= np.logspace(hmin, hmax, hres)
        fn = "networks/network_grid14.dill"

        if Train_save_objects:
            #Loop over a grid of hyperparameters and learning rates. Save objects
            networks = np.zeros((eres, hres), dtype=object)
            for i, eta in enumerate(eta_array):
                for j, hyperp in enumerate(hyperp_array):
                    n1 = logistic_NNW(X, y, Xt, yt, eta=eta, hyperp=hyperp)
                    networks[int(i),int(j)] = n1
                    print(f"eta = {eta}, hyperp = {hyperp} Completed.")

            with open(fn, "wb") as f:
                dill.dump(networks,f)
                print("Done")

        if Test_saved_objects:

            with open(fn, "rb") as f:
                networks = dill.load(f)

            results = 3 # look at the accuracy, the area under curve, and F1 score
            parameters = np.zeros((results, eres, hres))

            for i in range(eres):
                for j in range(hres):
                    n1 = networks[i,j]
                    if not n1: # some errors
                        pass
                    else:
                        n1.test_logistic_neuron(Xt, yt, load_data=False, \
                            cumulative_gain=True, confusion_matrix=True, \
                            network_summary=False)

                        #Accuracy saving
                        if hasattr(n1, 'acc'):
                            I = n1.acc
                            if str(I)=="nan":
                                parameters[0,i,j] = -1
                            else:
                                parameters[0,i,j] = I
                        else:
                            parameters[0,i,j] = -1

                        #Area Under Curve saving
                        if hasattr(n1, 'auc'):
                            A = n1.auc
                            if str(A)=="nan":
                                parameters[1,i,j] = -1
                            else:
                                parameters[1,i,j] = A
                        else:
                            parameters[1,i,j] = -1

                        #F1 Score saving
                        if hasattr(n1, 'F1'):
                            F = n1.F1
                            if str(F)=="nan":
                                parameters[2,i,j] = -1
                            else:
                                parameters[2,i,j] = F
                        else:
                            parameters[2,i,j] = -1

            # Produce heatmaps of the parameters:


            ax = sb.heatmap(data = pd.DataFrame(parameters[0]), annot=True,\
                vmin=0, vmax=1)
            bottom, top = ax.get_ylim()
            ax.set_ylim(bottom + 0.5, top - 0.5)    # Fix edges
            plt.title(rf"Accuracy scores, using arrays:" + "\n"+\
                rf"$\log\lambda_i \in"+rf"[{hmin:.0f}, {hmax:.0f}] $,"+\
                rf" resolution {hres:.0f}." + "\n" +\
                rf"$\log\eta_i \in"+rf"[{emin:.0f}, {emax:.0f}] $,"+\
                rf" resolution {eres:.0f}.")
            plt.xlabel(r"Hyperparameter index $i$")
            plt.ylabel(r"Learning rate index $i$")
            plt.figure()

            ax = sb.heatmap(data = pd.DataFrame(parameters[1]), annot=True,\
                vmin=0, vmax=1)
            bottom, top = ax.get_ylim()
            ax.set_ylim(bottom + 0.5, top - 0.5)
            plt.title(rf"Area under curve scores, using arrays:"+"\n"+\
                rf"$\log\lambda_i \in"+rf"[{hmin:.0f}, {hmax:.0f}] $,"+\
                rf" resolution {hres:.0f}." + "\n" +\
                rf"$\log\eta_i \in"+rf"[{emin:.0f}, {emax:.0f}] $,"+\
                rf" resolution {eres:.0f}.")
            plt.xlabel(r"Hyperparameter index $i$")
            plt.ylabel(r"Learning rate index $i$")
            plt.figure()

            ax = sb.heatmap(data = pd.DataFrame(parameters[2]), annot=True,\
                vmin=0, vmax=1)
            bottom, top = ax.get_ylim()
            ax.set_ylim(bottom + 0.5, top - 0.5)
            plt.title(rf"F1 scores, using arrays:" + "\n"+\
                rf"$\log\lambda_i \in"+rf"[{hmin:.0f}, {hmax:.0f}] $,"+\
                rf" resolution {hres:.0f}." + "\n" +\
                rf"$\log\eta_i \in"+rf"[{emin:.0f}, {emax:.0f}] $,"+\
                rf" resolution {eres:.0f}.")
            plt.xlabel(r"Hyperparameter index $i$")
            plt.ylabel(r"Learning rate index $i$")
            plt.show()

        # print(Tensorflow_neural_network(X, y, Xt, yt)) # comparison with tensorflow

def linear_regression_study():
    # ----------------- Regression (franke function data) ----------------
    sd = int(time.time())
    # generate lots of data f(x, y) = z:
    resolution=100
    x = np.linspace(0,1,resolution)
    y = np.linspace(0,1,resolution)
    X, Y = np.meshgrid(x, y)
    z = fns.franke_function(X, Y)



    # Generate the dataset as a (N x 2) matrix X and (N x 1) vector y.
    Xf = np.zeros((X.size, 2))
    Xf[:,0] = X.flatten()
    Xf[:,1] = Y.flatten()
    yf = z.flatten()[:,np.newaxis]

    X, Xt, y, yt = sklearn.model_selection.train_test_split(
        Xf, yf, test_size=0.1, random_state=sd)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, z)
    # ax.set_xlabel(r"x")
    # ax.set_ylabel(r"y")
    # ax.set_zlabel(r"$f(x, y)$")
    # plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(Xt[:,0], Xt[:,1], yt[:,0], "ro")
    # ax.set_xlabel(r"x")
    # ax.set_ylabel(r"y")
    # ax.set_zlabel(r"$f(x, y)$")
    # plt.show()

    gridsearch = 0
    singlesearch = 1

    if gridsearch:
        emin = 0; emax = -1; eres = 2  # resolution and limits of search
        hmin = 0; hmax = -1; hres = 2
        eta_array   = np.logspace(emin, emax, eres)
        hyperp_array= np.logspace(hmin, hmax, hres)
        MSE_grid = np.zeros((eres, hres))

        for i, eta in enumerate(eta_array):
            for j, hyperp in enumerate(hyperp_array):
                n1 = regression_NNW(X, y, Xt, yt, eta=eta, hyperp=hyperp)
                #   ----Produce an output using this trained neuron----
                n1.test_regression_neuron(Xt, yt, load_data=False, cfn=' ')
                n1.cost = n1.cost_fn(n1.ypred)
                MSE_grid[int(i),int(j)] = n1.cost
                print(f"eta = {eta}, hyperp = {hyperp} Completed.")


        ax = sb.heatmap(data = pd.DataFrame(MSE_grid), annot=True)#, vmin=0, vmax=1)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)    # Fix edges
        plt.title(rf"MSE Scores, using arrays:" + "\n"+\
            rf"$\log\lambda_i \in"+rf"[{hmin:.0f}, {hmax:.0f}] $,"+\
            rf" resolution {hres:.0f}." + "\n" +\
            rf"$\log\eta_i \in"+rf"[{emin:.0f}, {emax:.0f}] $,"+\
            rf" resolution {eres:.0f}.")
        plt.xlabel(r"Hyperparameter index $i$")
        plt.ylabel(r"Learning rate index $i$")
        plt.figure()

    elif singlesearch:
        eta = 0.001
        hyperp = 0
        n1 = regression_NNW(X, y, Xt, yt, eta=eta, hyperp=hyperp)
          # ----Produce an output using this trained neuron----
        n1.test_regression_neuron(Xt, yt, load_data=False, cfn=' ')
        n1.cost = n1.cost_fn(n1.ypred)
        n1.regression_network_summary()

        # print(n1.ypred)
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        if n1.rests!=0:
            ax.plot(Xt[:,0], Xt[:,1], n1.output, "r.")
        else:
            ax.plot(Xt[:resolution,0], Xt[:resolution,1], n1.output.reshape(-1)[:resolution], "r.")
        ax.set_xlabel(r"x")
        ax.set_ylabel(r"y")
        ax.set_zlabel(r"$f(x, y)$")
        ax.set_title(r"Franke's Function $\hat{y}$")
        ax = fig.add_subplot(122, projection='3d')
        ax.plot(X[:,0], X[:,1], y[:,0], "b.")
        ax.set_xlabel(r"x")
        ax.set_ylabel(r"y")
        ax.set_zlabel(r"$f(x, y)$")
        ax.set_title(r"Franke's Function $y$")
        plt.show()

def Sklearn_sgd_classifier(X, y, Xt, yt):
    """
    Classification
    Logistic Regression
    Scikit-learns module: SGDClassifier
    """
    print("-------------------")
    solution = sklearn.linear_model.SGDClassifier(eta0=0.01, max_iter=100).fit(X, y)
    yp = solution.predict(X)
    # print(yp)
    # a = fns.assert_binary_accuracy(y, yp)
    if len(y)>len(yp):
        y = y[:len(yp)]
    elif len(y)<len(yp):
        yp = yp[:len(y)]
    fns.produce_confusion_mtx(y, yp)
    fns.produce_cgchart(y, yp)
    # return f"Sklearn\'s SGDClassifier accuracy: {100*a:.0f} %"

def SGD_with_minibatches(X, y, Xt, yt, gamma, max_iter, batch_size, verbose=False):
    """
    Classification
    Logistic Regression
    Our own SGD with mini-batches
    (see "./lib/logistic_regressionk.py")
    """
    obj = lgr.StochasticGradientDescent(gamma, max_iter, batch_size, \
        verbose=verbose)
    obj.fit(X, y)
    yp = obj.predict(Xt)
    # print(yp)
    # a = fns.assert_binary_accuracy(y, yp)
    #round:
    yp[np.where(yp<0.5)]=0
    yp[np.where(yp>=0.5)]=1
    if len(y)>len(yp):
        y = y[:len(yp)]
    elif len(y)<len(yp):
        yp = yp[:len(y)]
    fns.produce_confusion_mtx(y, yp)
    fns.produce_cgchart(y, yp)
    # return f"SGD with mini-batches accuracy: {100*a:.0f} %"

def Tensorflow_neural_network(X, y, Xt, yt):
    """
    Classification
    Neural Networks
    Tensorflows module
    """
    print("-------------------")
    yp = fns.tensorflow_NNWsolver(X, y, Xt, yt)
    # print(yp)
    a = fns.assert_binary_accuracy(yt, yp)
    return f"Tensorflow NN accuracy: {100*a:.0f} %"

def logistic_NNW(X, y, Xt, yt, eta=0.1, hyperp=0.1):
    """
    Classification
    Neural Networks
    Our own FFNN by using the backpropagation algorithm
    (see "./lib/neural_network.py")
    """
    n1 = nnw.Neuron(
        eta=eta,
        biases_str="xavier",
        weights_str="xavier",
        cost_fn_str="xentropy",
        batchsize=100,
        epochs=10
    )

    n1.features = X.shape[1]
    act = "relu6"
    n1.add_hlayer(55, activation=act)
    n1.add_hlayer(44, activation=act)
    n1.add_hlayer(33, activation=act)
    n1.set_outputs(y[0], activation="sigmoid")
    n1.set_inputs(X[0], init=True)
    n1.set_biases()
    n1.set_weights()
    n1.set_cost_fn(reg_str = ' ', hyperp=hyperp)  # requires in/outputs

    n1.train_neuron(X, y, save_network=False, verbose=False)
    n1.test_logistic_neuron(Xt, yt, load_data=False, cfn=' ', batchsize=100)
    plt.show()
    return n1

def regression_NNW(X, y, Xt, yt, eta=0.1, hyperp=0.1):
    """
    Classification
    Neural Networks
    Our own FFNN by using the backpropagation algorithm
    (see "./lib/neural_network.py")
    """
    n1 = nnw.Neuron(
        eta=eta,
        biases_str="xavier",
        weights_str="xavier",
        cost_fn_str="MSE",
        batchsize=100,
        epochs=200
    )

    n1.features = X.shape[1]
    act = "relu"
    n1.add_hlayer(10, activation=act)
    n1.add_hlayer(10, activation=act)
    n1.set_outputs(y[0], activation="identity")
    n1.set_inputs(X[0], init=True)
    n1.set_biases()
    n1.set_weights()
    n1.set_cost_fn(reg_str = ' ', hyperp=hyperp)  # require in/outputs

    n1.train_neuron(X, y, save_network=False)

    return n1

if __name__ == "__main__":
    start = time.time()
    main()
    print("-------------------")
    print(f"Completed in {time.time() - start:.2f} seconds.")

"""
Neural network slides:
https://compphysics.github.io/MachineLearning/doc/pub/NeuralNet/html/._NeuralNet-bs023.html
"""
