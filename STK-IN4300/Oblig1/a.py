from sklearn.preprocessing import PolynomialFeatures as Poly
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold
import rpy2.robjects as robjects
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np

from warnings import filterwarnings
filterwarnings('ignore')

def calculate(data):
    MSE_lasso_step = []
    MSE_ridge_step = []
    a = data[0]
    max_iter = data[1]

    for X_train, X_test, y_train, y_test in data[2]:

        lasso = Lasso(alpha = a, max_iter = max_iter)
        lasso.fit(X_train, y_train)

        ridge = Ridge(alpha = a)
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
N = 1E3
alphas = np.logspace(-8, 3, N)
max_iter_vals = [5, 1000]
filenames = ["a_5_iter", "a"]

kf = KFold(n_splits = k)
sets = []
poly = Poly(degree = degree)
for train_index, test_index in kf.split(X):
    sets.append((poly.fit_transform(X[train_index]),
    poly.fit_transform(X[test_index]), y[train_index], y[test_index]))

for max_iter, filename in zip(max_iter_vals, filenames):

    MSE_lasso = np.zeros_like(alphas)
    MSE_ridge = np.zeros_like(alphas)

    pool = Pool()
    n = 0
    for i in pool.imap(calculate, ([alphas[j], max_iter, sets] for j in range(int(N)))):
        MSE_lasso[n], MSE_ridge[n] = i
        n += 1
        print(f"\r{100*n/len(alphas):.2f}", end = "")
    print()

    argmin_lasso = np.argmin(MSE_lasso)
    argmin_ridge = np.argmin(MSE_ridge)

    print(f"LASSO lambda = {alphas[argmin_lasso]}, MSE = {MSE_lasso[argmin_lasso]}")
    print(f"Ridge lambda = {alphas[argmin_ridge]}, MSE = {MSE_ridge[argmin_ridge]}")

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
