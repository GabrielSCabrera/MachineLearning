from sklearn.preprocessing import PolynomialFeatures as Poly
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import normalize
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import KFold
import rpy2.robjects as robjects
import matplotlib.pyplot as plt
import numpy as np

from warnings import filterwarnings
filterwarnings('ignore')

def calculate(data):
    MSE_lasso_step = []
    MSE_ridge_step = []
    a = data[0]
    max_iter = data[1]

    lasso = Lasso(alpha = a, max_iter = max_iter, precompute = True)
    ridge = Ridge(alpha = a)

    for X_train, X_test, y_train, y_test in data[2]:

        lasso.fit(X_train, y_train)
        ridge.fit(X_train, y_train)

        y_lasso = lasso.predict(X_test)
        y_ridge = ridge.predict(X_test)

        MSE_lasso_step.append(np.mean((y_lasso - y_test)**2))
        MSE_ridge_step.append(np.mean((y_ridge - y_test)**2))

    return np.mean(MSE_lasso_step), np.mean(MSE_ridge_step)

robjects.r['load']("data_o1.rdata")
X = np.array(robjects.r['X'])
y =  np.array(robjects.r['y'])

X = (X - np.mean(X))/np.std(X)
y = (y - np.mean(y))/np.std(y)

k = 10
degree = 1
N = 5E2
alphas = np.logspace(-8, 4, N)
max_iter_vals = [10, 100, 1000]
outputfile = "outputdata.txt"
filenames = ["a_10_iter", "a_100_iter", "a_1000_iter"]

kf = KFold(n_splits = k)
sets = []
poly = Poly(degree = degree)
for train_index, test_index in kf.split(X):
    sets.append((poly.fit_transform(X[train_index]),
    poly.fit_transform(X[test_index]), y[train_index], y[test_index]))

outfile = open(outputfile, "w+")
pool = Pool()

for max_iter, filename in zip(max_iter_vals, filenames):

    MSE_lasso = np.zeros_like(alphas)
    MSE_ridge = np.zeros_like(alphas)

    n = 0
    for i in pool.imap(calculate, ([alphas[j], max_iter, sets] for j in range(int(N)))):
        MSE_lasso[n], MSE_ridge[n] = i
        n += 1
        print(f"\r{100*n/len(alphas):.2f}", end = "")
    print()

    argmin_lasso = np.argmin(MSE_lasso)
    argmin_ridge = np.argmin(MSE_ridge)

    msg = (f"For max_iter = {max_iter}, we have:\n"
           f"LASSO lambda = {alphas[argmin_lasso]}, MSE = {MSE_lasso[argmin_lasso]}\n"
           f"Ridge lambda = {alphas[argmin_ridge]}, MSE = {MSE_ridge[argmin_ridge]}\n")

    outfile.write(msg)
    print(msg)
    plotfunc = plt.semilogx

    plotfunc(alphas, MSE_lasso, "b-", label = "LASSO")
    plotfunc(alphas, MSE_ridge, "r-", label = "Ridge")
    plotfunc([alphas[argmin_lasso]], [MSE_lasso[argmin_lasso]], "bx", label = "LASSO minimum")
    plotfunc([alphas[argmin_ridge]], [MSE_ridge[argmin_ridge]], "rx", label = "Ridge Minimum")
    plt.xlabel("Hyperparameter $\\lambda$")
    plt.ylabel("$MSE$")
    plt.legend()
    plt.xlim([np.min(alphas), np.max(alphas)])
    plt.savefig(f"{filename}.png", dpi = 250)
    plt.close()

    plotfunc(alphas, MSE_lasso, "b-", label = "Lasso")
    plotfunc(alphas, MSE_ridge, "r-", label = "Ridge")
    plotfunc([alphas[argmin_lasso]], [MSE_lasso[argmin_lasso]], "bx", label = "LASSO minimum")
    plotfunc([alphas[argmin_ridge]], [MSE_ridge[argmin_ridge]], "rx", label = "Ridge Minimum")
    plt.xlabel("Hyperparameter $\\lambda$")
    plt.ylabel("$MSE$")
    plt.legend()
    plt.xlim([np.min(alphas), np.max(alphas)])
    plt.savefig(f"{filename}.pdf", dpi = 250)
    plt.close()

outfile.close()
