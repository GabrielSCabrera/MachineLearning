import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import subprocess, os, sys
from matplotlib import cm
from time import time
import pandas as pd
import numpy as np

np.seterr("ignore")

sys.path.append("..")
from backend.neuralnet import NeuralNet, split

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
    precision = np.sum(TP_arr)/(np.sum(TP_arr)+np.sum(FP_arr))
    recall = np.sum(TP_arr)/(np.sum(TP_arr)+np.sum(FN_arr))

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

def parse_args(all_args):
    N_args = len(sys.argv)
    sp = 20
    valid_args = f"\n{'keywords':^{sp}s}{'type':^{sp}s}{'value':^{sp}s}\n"
    valid_args += "-"*len(valid_args) + "\n"
    valid_args = "\n\nValid options are as follows:\n" + valid_args
    for key, val in all_args.items():
        valid_args += f"{key:^{sp}s}{str(val[0]):^{sp}s}{val[1]:^{sp}s}\n"

    if N_args > 1:
        args = sys.argv[1:]
        prev_keys = []
        prev_vals = []
        prev_locs = []
        for arg in args:
            arg = arg.strip().split("=")
            # arg = arg.lower().strip().split("=")
            if len(arg) != 2:
                msg = (f"\n\nInvalid command-line argument format.  Keyword "
                       f"arguments are expected.  Attempted to pass {len(arg)}"
                       f" term(s) in one argument\n\t{arg}")
                msg += valid_args
                raise ValueError(msg)

            key = arg[0].strip()
            val = arg[1].strip()
            if key in prev_keys:
                msg = (f"\n\nAttempting to pass a command-line argument "
                       f"multiple times.\n\tkeyword:\t{key}\n\t  value:\t{val}")
                msg += valid_args
                raise NameError(msg)
            elif key in all_args.keys():
                new_type = all_args[key][0]
                msg = (f"\n\nAttempting to pass an argument of invalid "
                f"type.\n\tkeyword:\t{key}\n\t  value:\t{val}"
                f"\n\texpects:\t{all_args[key][0]}")
                msg += valid_args
                try:
                    val = new_type(val)
                except ValueError:
                    raise ValueError(msg)
                except TypeError:
                    raise TypeError(msg)
                prev_keys.append(key)
                prev_vals.append(val)
                prev_locs.append(all_args[key][1])
            else:
                msg = (f"\n\nAttempting to pass invalid command-line argument"
                       f"\n\tkeyword:\t{key}\n\t  value:\t{val}")
                msg += valid_args
                raise NameError(msg)
        for key,val,loc in zip(prev_keys, prev_vals, prev_locs):
            globals()[loc] = val

time_0 = time()
grid_size = 10

regs1 = np.logspace(-8, -6, grid_size)
lrs1 = np.linspace(0.06, 0.2, grid_size)

regs2 = np.logspace(-6, -5, grid_size)
lrs2 = np.linspace(0.08, 0.2, grid_size)

epochs = 200
gpu = False
dir1 = "grid_cc"
dir2 = "grid_franke"
test_percent = 25
# Gaussian noise in Franke function
sigma = 0.1
# Activation function
activation_fxn = "sigmoid"
# Activation function for output layer (None defaults to "activation_fxn")
output_activation_fxn = None
# Random Seed
rand_seed = 112358

all_args = {"gridsize":[int, "grid_size"], "epochs":[int, "epochs"],
            "gpu":[bool, "gpu"], "dir1":[str, "dir1"], "dir2":[str, "dir2"],
            "test_percent":[float, "test_percent"], "sigma":[float, "sigma"],
            "activation_fxn":[str, "activation_fxn"],
            "output_activation_fxn":[str, "output_activation_fxn"],
            "rand_seed":[int, "rand_seed"], "mode":[str, "cmdlinearg"]}

parse_args(all_args)

if output_activation_fxn is None:
    output_activation_fxn = activation_fxn

np.random.seed(rand_seed)

if cmdlinearg == "write":

    def execute(lr, reg, dirname, epochs, gpu, filename, lrcount, regcount, savename):
        loc = f"{dirname}/lr_{lrcount}_reg_{regcount}"
        try:
            os.mkdir(loc)
        except FileExistsError:
            pass
        subprocess.call(['python3', filename,
                        f'save={loc}', f'lr={lr}', f'saveimg={savename}',
                        f'epochs={epochs}', f"reg={reg}"])#, f"GPU={gpu}"])
    try:
        os.mkdir(dir1)
    except FileExistsError:
        pass

    np.save(f"{dir1}/regs", regs1)
    np.save(f"{dir1}/lrs", lrs1)

    try:
        os.mkdir(dir2)
    except FileExistsError:
        pass

    np.save(f"{dir2}/regs", regs2)
    np.save(f"{dir2}/lrs", lrs2)

    for m,lr in enumerate(lrs1):
        for n,reg in enumerate(regs1):
            execute(lr, reg, dir1, epochs, gpu, 'credit_card.py',m,n, "roc.png")

    # for m,lr in enumerate(lrs2):
    #     for n,reg in enumerate(regs2):
    #         execute(lr, reg, dir2, epochs, gpu, 'franke.py',m,n, "franke.png")

if cmdlinearg in ["write", "read"]:

    regs1 = np.load(f"{dir1}/regs.npy")
    lrs1 = np.load(f"{dir1}/lrs.npy")
    regs2 = np.load(f"{dir2}/regs.npy")
    lrs2 = np.load(f"{dir2}/lrs.npy")

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

    del X_train
    del Y_train

    print("\rReading Credit Card Data   [DONE]\n\nGenerating Franke Data...", end = "")

    X, Y, f_xy = generate_Franke_data(N = 150)
    X_train, Y_train, X_test_franke, Y_test_franke = split(X, Y, test_percent)

    del X_train
    del Y_train

    print("\rGenerating Franke Data   [DONE]")

    NN = NeuralNet()

    dirs1_unprocessed = np.sort(os.listdir(dir1))
    dirs2_unprocessed = np.sort(os.listdir(dir2))

    dirs1 = []
    dirs2 = []

    for n,d1 in enumerate(dirs1_unprocessed):
        if "." not in d1:
            dirs1.append(d1)

    for n,d2 in enumerate(dirs2_unprocessed):
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
        Y_predict_cc, Y_test_cc2 = NN.predict(X_test_cc, Y_test_cc)
        acc_cc[arg_map[n1]] = accuracy(Y_predict_cc, Y_test_cc2)
        AUC_cc[arg_map[n1]], F1_cc[arg_map[n1]] = F1_AUC(Y_predict_cc, Y_test_cc2)
        gains_cc[arg_map[n1]] = cum_gains(Y_predict_cc, Y_test_cc2)
        perc1 = int(100*n1/tot1)
        print(f"\rProcessing Credit Card Data:\t{perc1:4d}%", end = "")
    print(f"\rProcessing Credit Card Data:\t{100:4d}%")

    np.save(f"{dir1}/acc", acc_cc)
    np.save(f"{dir1}/F1", F1_cc)
    np.save(f"{dir1}/AUC", AUC_cc)
    np.save(f"{dir1}/gains", gains_cc)

    MSE_franke = np.zeros((grid_size, grid_size))
    R2_franke = np.zeros((grid_size, grid_size))

    print(f"\rProcessing Franke Data:\t\t{0:4d}%", end = "")
    tot2 = len(dirs2)
    for n2,d2 in enumerate(dirs2):
        NN.set(X_test_franke, Y_test_franke)
        NN.load(f"{dir2}/{d2}")
        Y_predict_franke, Y_test_franke2 = NN.predict(X_test_franke, Y_test_franke)
        MSE_franke[arg_map[n2]] = MSE(Y_predict_franke, Y_test_franke2)
        R2_franke[arg_map[n2]] = R2(Y_predict_franke, Y_test_franke2)
        perc2 = int(100*n2/tot2)
        print(f"\rProcessing Franke Data:\t\t{perc2:4d}%", end = "")
    print(f"\rProcessing Franke Data:\t\t{100:4d}%")

    np.save(f"{dir2}/MSE", MSE_franke)
    np.save(f"{dir2}/R2", R2_franke)

elif cmdlinearg == "plot":

    acc_cc = np.load(f"{dir1}/acc.npy")
    F1_cc = np.load(f"{dir1}/F1.npy")
    AUC_cc = np.load(f"{dir1}/AUC.npy")
    gains_cc = np.load(f"{dir1}/gains.npy")

    regs1 = np.load(f"{dir1}/regs.npy")
    lrs1 = np.load(f"{dir1}/lrs.npy")

    X1,Y1 = np.meshgrid(lrs1, regs1)

    cmap = cm.magma

    labels1 = ["Accuracy-Score", "F1-Score", "AUC-Score", "Gains-Ratio"]
    values1 = [acc_cc, F1_cc, AUC_cc, gains_cc]

    x1_diffs = np.diff(X1, axis = 1)
    x1_text = X1.copy()[:,:-1]
    x1_text = x1_text + x1_diffs/9

    y1_diffs = np.diff(Y1, axis = 0)
    y1_text = Y1.copy()[:-1]
    y1_text = y1_text + y1_diffs/4.25

    for label, value in zip(labels1, values1):
        heatmap = plt.pcolormesh(X1, Y1, value, cmap=cmap)
        cbar = plt.colorbar(heatmap)
        cbar.ax.set_ylabel(label)
        for i,j,k in zip(x1_text, y1_text, value):
            for x,y,z in zip(i,j,k):
                val = "." + f"{z:.3f}".split(".")[1]
                txt = plt.text(x, y, val, weight = 'bold')
                txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
        plt.yscale('log')
        plt.xlabel("Learning Rate $\eta$")
        plt.ylabel("Regularization Parameter $\lambda$")
        plt.show()

    MSE_franke = np.load(f"{dir2}/MSE.npy")
    R2_franke = np.load(f"{dir2}/R2.npy")

    regs2 = np.load(f"{dir2}/regs.npy")
    lrs2 = np.load(f"{dir2}/lrs.npy")

    X2,Y2 = np.meshgrid(lrs2, regs2)

    labels2 = ["MSE", "R²-Score"]
    values2 = [MSE_franke, R2_franke]

    x2_diffs = np.diff(X2, axis = 1)
    x2_text = X2.copy()[:,:-1]
    x2_text = x2_text + x2_diffs/9

    y2_diffs = np.diff(Y2, axis = 0)
    y2_text = Y2.copy()[:-1]
    y2_text = y2_text + y2_diffs/4.25

    for label, value in zip(labels2, values2):
        heatmap = plt.pcolormesh(X2, Y2, value, cmap=cmap)
        cbar = plt.colorbar(heatmap)
        cbar.ax.set_ylabel(label)
        for i,j,k in zip(x2_text, y2_text, value):
            for x,y,z in zip(i,j,k):
                if "." in f"{z:.3f}":
                    val = "." + f"{z:.3f}".split(".")[1]
                else:
                    val = z
                txt = plt.text(x, y, val, weight = 'bold')
                txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
        plt.yscale('log')
        plt.xlabel("Learning Rate $\eta$")
        plt.ylabel("Regularization Parameter $\lambda$")
        plt.show()

else:
    raise NotImplementedError("Must choose one of the following cmdline arguments:\n\twrite\n\tread\n\tplot")

dt = time() - time_0
hh = int(dt//3600)
mm = int((dt//60)%60)
ss = int(dt%60)
print(f"\n\nTotal Time Elapsed {hh:02d}:{mm:02d}:{ss:02d}")
