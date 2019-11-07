import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
from matplotlib import cm
import subprocess, os
import pandas as pd
import numpy as np
import sys

np.seterr("ignore")

sys.path.append("..")
from backend.neuralnet import NeuralNet, preprocess, split

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def generate_Franke_data(x_min = 0, x_max = 1, N = 250):

    # Generating NxN meshgrid of x,y values in range [x_min, x_max]
    X = np.random.random((N,N))*(x_max-x_min) + x_min
    Y = np.random.random((N,N))*(x_max-x_min) + x_min

    # Calculating the values of the Franke function at each (x,y) coordinate
    Z = FrankeFunction(X,Y)
    init_error = np.random.normal(0, globals()["sigma"], Z.shape)
    f_xy = Z.copy().flatten()
    Z = Z + init_error

    # Making compatible input arrays for Regression object
    x = np.zeros((X.size, 2))
    x[:,0] = X.flatten()
    x[:,1] = Y.flatten()
    y = Z.reshape((Z.size, 1))
    # y = Z.flatten()[:,np.newaxis]

    return x, y, f_xy

def MSE(Y_predict, Y_test):
    return np.mean((Y_predict - Y_test)**2)

def R2(Y_predict, Y_test):
    SS_tot = np.sum((Y_test-np.mean(Y_test))**2)
    SS_res = np.sum((Y_predict-Y_test)**2)
    return 1 - SS_res/SS_tot

def F1_AUC(Y_predict, Y_test):
    Yf_test = Y_test.flatten()
    Yf_predict = Y_predict.flatten()
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
    FN_arr = np.logical_and(np.equal(Yf_predict, np.logical_not(Yf_test)),
                            np.equal(Yf_test, 1))
    FP_arr = np.logical_and(np.equal(Yf_predict, np.logical_not(Yf_test)),
                            np.equal(Yf_test, 0))

    recall = 0
    precision = 0
    tpr = np.zeros_like(Yf_test, dtype = np.float64)
    fpr = np.zeros_like(Yf_test, dtype = np.float64)

    denom1 = TP_arr + FN_arr
    denom2 = FP_arr + TN_arr
    denom3 = TP_arr + FP_arr
    denom1[denom1 == 0] = np.inf
    denom2[denom2 == 0] = np.inf
    denom3[denom3 == 0] = np.inf
    tpr = np.cumsum(TP_arr/denom1)
    fpr = np.cumsum(FP_arr/denom2)
    recall = np.sum(TP_arr/denom1)
    precision = np.sum(TP_arr/denom3)

    # Normalization constant
    norm = 1

    tpr = norm*(tpr - np.min(tpr))/(np.max(tpr) - np.min(tpr))
    fpr = norm*(fpr - np.min(fpr))/(np.max(fpr) - np.min(fpr))

    AUC = np.trapz(tpr, fpr)
    F1 = 2*(precision*recall)/(precision + recall)

    return AUC, F1

def cum_gains(Y_predict, Y_test):
    Yf_test = Y_test.flatten()
    Yf_predict = Y_predict.flatten()
    idx1 = np.argsort(Yf_predict)[::-1]
    idx2 = np.argsort(Yf_test)[::-1]

    Yf_test1 = Yf_test[idx1]
    Yf_test2 = Yf_test[idx2]

    gains1 = np.cumsum(Yf_test1)/np.sum(Yf_test1)
    percent1 = np.arange(1, len(Yf_test1)+1)/len(Yf_test1)

    gains2 = np.cumsum(Yf_test2)/np.sum(Yf_test2)
    percent2 = np.arange(1, len(Yf_test2)+1)/len(Yf_test2)

    A1 = np.trapz(gains1, percent1)
    A2 = np.trapz(gains2, percent2)
    A3 = np.trapz(percent1, percent1)

    score = (A1-A3)/(A2-A3)

    # plt.plot(percent1, gains1)
    # plt.plot(percent2, gains2)
    # plt.plot(percent1, percent1)
    # plt.show()
    # exit()

    return score

def accuracy(Y_predict, Y_test):
    diffs = Y_predict-Y_test
    incorrect = np.sum(np.abs(diffs))
    correct = diffs.shape[0] - incorrect
    total = diffs.shape[0]
    return correct/total

grid_size = 10
regs = np.logspace(-8, -2, grid_size)
lrs = np.linspace(0.005, 0.2, grid_size)
epochs = 200
gpu = True
dir1 = "grid_cc"
dir2 = "grid_franke"
test_percent = 25
# Gaussian noise in Franke function
sigma = 0.1
# Activation function
activation_fxn = "sigmoid"
# Activation function for output layer (None defaults to "activation_fxn")
output_activation_fxn = None
rand_seed = 112358

if output_activation_fxn is None:
    output_activation_fxn = activation_fxn

np.random.seed(rand_seed)

if len(sys.argv) == 1:
    cmdlinearg = None
else:
    cmdlinearg = sys.argv[1]

if cmdlinearg == "write":

    def execute(lr, reg, dirname, epochs, gpu, filename, lrcount, regcount, savename):
        loc = f"{dirname}/lr_{lrcount}_reg_{regcount}"
        try:
            os.mkdir(loc)
        except FileExistsError:
            pass
        subprocess.call(['python3', filename,
                        f'save={loc}', f'lr={lr}', f'saveimg={savename}',
                        f'epochs={epochs}', f"reg={reg}", f"GPU={gpu}"])
    try:
        os.mkdir(dir1)
    except FileExistsError:
        pass

    np.save(f"{dir1}/regs", regs)
    np.save(f"{dir1}/lrs", lrs)

    try:
        os.mkdir(dir2)
    except FileExistsError:
        pass

    np.save(f"{dir2}/regs", regs)
    np.save(f"{dir2}/lrs", lrs)

    for m,lr in enumerate(lrs):
        for n,reg in enumerate(regs):
            execute(lr, reg, dir1, epochs, gpu, 'credit_card.py',m,n, "roc.png")
            execute(lr, reg, dir2, epochs, gpu, 'franke.py',m,n, "franke.png")

elif cmdlinearg == "read":

    print("Reading Credit Card Data...", end = "")

    filename = "ccdata.xls"
    df = pd.read_excel(filename)
    X_labels = list(df.iloc()[0,:-1])
    Y_label = list(df.iloc()[0,-1])
    X = np.array(df.iloc()[1:,1:-1], dtype = np.int64)
    Y = np.array(df.iloc()[1:,-1], dtype = np.int64)[:,np.newaxis]
    del df

    categorical_cols = {"1":[1,2], "2":[1,2,3,4], "3":[1,2,3]}
    X_train, Y_train, X_test_cc, Y_test_cc = split(X, Y, test_percent)
    X_test_cc, Y_test_cc = preprocess(X_test_cc, Y_test_cc, categorical_cols,
                                      True, output_activation_fxn)
    del X_train
    del Y_train

    print("\rReading Credit Card Data   [DONE]\n\nGenerating Franke Data...", end = "")

    X, Y, f_xy = generate_Franke_data(N = 150)
    X_train, Y_train, X_test_franke, Y_test_franke = split(X, Y, test_percent)
    X_test_franke, Y_test_franke = preprocess(X_test_franke, Y_test_franke, {},
                                              True, output_activation_fxn)
    del X_train
    del Y_train

    print("\rGenerating Franke Data   [DONE]")

    NN = NeuralNet()

    dirs1_unprocessed = np.sort(os.listdir(dir1))
    dirs2_unprocessed = np.sort(os.listdir(dir2))

    dirs1 = []
    dirs2 = []

    for n,(d1,d2) in enumerate(zip(dirs1_unprocessed, dirs2_unprocessed)):
        if "." not in d1:
            dirs1.append(d1)
        if "." not in d2:
            dirs2.append(d2)

    arg_map = [(i,j) for i in range(grid_size) for j in range(grid_size)]

    acc_cc = np.zeros((grid_size, grid_size))
    F1_cc = np.zeros((grid_size, grid_size))
    AUC_cc = np.zeros((grid_size, grid_size))
    gains_cc = np.zeros((grid_size, grid_size))

    print(f"Processing Credit Card Data:\t{0:4d}%", end = "")
    tot1 = len(dirs1)
    for n1,d1 in enumerate(dirs1):
        NN.set(X_test_cc, Y_test_cc)
        NN.load(f"{dir1}/{d1}")
        Y_predict_cc = NN.predict(X_test_cc)
        acc_cc[arg_map[n1]] = accuracy(Y_predict_cc, Y_test_cc)
        AUC_cc[arg_map[n1]], F1_cc[arg_map[n1]] = F1_AUC(Y_predict_cc, Y_test_cc)
        gains_cc[arg_map[n1]] = cum_gains(Y_predict_cc, Y_test_cc)
        perc1 = int(100*n1/tot1)
        print(f"\rProcessing Credit Card Data:\t{perc1:4d}%", end = "")
    print(f"\rProcessing Credit Card Data:\t{100:4d}%")

    MSE_franke = np.zeros((grid_size, grid_size))
    R2_franke = np.zeros((grid_size, grid_size))

    print(f"\rProcessing Franke Data:\t\t{0:4d}%", end = "")
    tot2 = len(dirs2)
    for n2,d2 in enumerate(dirs2):
        NN.set(X_test_franke, Y_test_franke)
        NN.load(f"{dir2}/{d2}")
        Y_predict_franke = NN.predict(X_test_franke)
        MSE_franke[arg_map[n2]] = MSE(Y_predict_franke, Y_test_franke)
        R2_franke[arg_map[n2]] = R2(Y_predict_franke, Y_test_franke)
        perc2 = int(100*n2/tot2)
        print(f"\rProcessing Franke Data:\t\t{perc2:4d}%", end = "")
    print(f"\rProcessing Franke Data:\t\t{100:4d}%")

    np.save(f"{dir1}/acc", acc_cc)
    np.save(f"{dir1}/F1", F1_cc)
    np.save(f"{dir1}/AUC", AUC_cc)
    np.save(f"{dir1}/gains", gains_cc)

    np.save(f"{dir2}/MSE", MSE_franke)
    np.save(f"{dir2}/R2", R2_franke)

elif cmdlinearg == "plot":

    acc_cc = np.load(f"{dir1}/acc.npy")
    F1_cc = np.load(f"{dir1}/F1.npy")
    AUC_cc = np.load(f"{dir1}/AUC.npy")
    gains_cc = np.load(f"{dir1}/gains.npy")

    MSE_franke = np.load(f"{dir2}/MSE.npy")
    R2_franke = np.load(f"{dir2}/R2.npy")

    X,Y = np.meshgrid(lrs, regs)

    cmap = cm.magma

    labels = ["Accuracy-Score", "F1-Score", "AUC-Score", "Gains-Ratio", "MSE", "R²-Score"]
    values = [acc_cc, F1_cc, AUC_cc, gains_cc, MSE_franke, R2_franke]

    x_diffs = np.diff(X, axis = 1)
    x_text = X.copy()[:,:-1]
    x_text = x_text + x_diffs/9

    y_diffs = np.diff(Y, axis = 0)
    y_text = Y.copy()[:-1]
    y_text = y_text + y_diffs/4.25

    for label, value in zip(labels, values):
        heatmap = plt.pcolormesh(X, Y, value, cmap=cmap)
        cbar = plt.colorbar(heatmap)
        cbar.ax.set_ylabel(label)
        for i,j,k in zip(x_text, y_text, value):
            for x,y,z in zip(i,j,k):
                val = "." + f"{z:.3f}".split(".")[1]
                txt = plt.text(x, y, val, weight = 'bold')
                txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
        plt.yscale('log')
        plt.xlabel("Learning Rate $\eta$")
        plt.ylabel("Regularization Parameter $\lambda$")
        plt.show()
else:
    raise NotImplementedError("Must choose one of the following cmdline arguments:\n\twrite\n\tread\n\tplot")
