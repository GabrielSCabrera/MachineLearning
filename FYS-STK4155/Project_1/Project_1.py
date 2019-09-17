import matplotlib.pyplot as plt
import numpy as np
import sys, franke

sys.path.append("..")
from utils.classes import Regression

plt.style.use("seaborn")

# Conditions
k = 5               # k in k-fold
degree = 5          # Polynomial approximation degree
sigma = 1           # Variance of Gaussian Noise
split_test = 20     # Percentage of data to split into testing set

"""PART A"""
print("\n" + "-"*80 + "\nPART A\n" + "-"*80)

# Select random seed for consistent results
np.random.seed(69420666)

# # Generating NxN meshgrid of x,y values in range [0, 1]
# x_min, x_max, N = 0, 1, 100
# x = np.linspace(x_min, x_max, int(N))
# X,Y = np.meshgrid(x, x)
#
# # Calculating the values of the Franke function at each (x,y) coordinate
# Z = franke.FrankeFunction(X,Y)
# Z = Z + np.random.normal(0, 0.1, Z.shape)
#

X = np.random.random(size = 10000)
Y = np.random.random(size = 10000)
Z = franke.FrankeFunction(X, Y) + np.random.normal(0, .01, size = X.shape[0])

# Making compatible input arrays for Regression object
# x = np.zeros((X.shape[0]*X.shape[1], 2))
x = np.zeros((X.shape[0], 2))
x[:,0] = X#.flatten()
x[:,1] = Y#.flatten()
y = Z.flatten()
#X = x

# Creating Regression object with x and y
R = Regression(x, y)

# Implementing 5th degree polynomial regression in 2-D
R.poly(degree = degree)


def part_a():
    # R.plot()

    # Creating <dict> of values for OLS
    OLS_data = {}

    # Calculating the variance in beta
    OLS_data["var"] = R.variance(sigma = 1)

    # Calculating the MSE
    OLS_data["MSE"] = R.mse()

    # Calculating the R-squared score
    OLS_data["R2"] = R.r_squared()

    sigma2 = R.sigma()

    # Displaying Results
    var = ", ".join(list(f"{i:.3g}" for i in OLS_data["var"]))
    print(f"\nVar(beta) = \n{var}")
    print(f"\nMSE = {OLS_data['MSE']:.2g}")
    print(f"\nR² = {OLS_data['R2']:.2g}")
    print(f"\nσ² = {sigma2}")

def part_b():


    """PART B"""
    print("\n" + "-"*80 + "\nPART B\n" + "-"*80)

    # Creating <dict> of values for OLS k-fold
    kfold_data = {}

    # Implementing the k-fold algorithm
    kfold_data["R2"], kfold_data["MSE"], kfold_data["var"] = \
    R.k_fold(k = k, degree = degree, sigma = sigma)

    # Displaying Results
    var = ", ".join(list(f"{i:.3g}" for i in kfold_data["var"]))
    print(f"\nVar(beta) = \n{var}")
    print(f"\nMSE = {kfold_data['MSE']:.2g}")
    print(f"\nR² = {kfold_data['R2']:.2g}")

def part_d():

    """PART D"""
    print("\n" + "-"*80 + "\nPART D\n" + "-"*80)

    # Creating <dict> of values for ridge regression
    ridge_data = {"ridge":{}, "k_fold":{}}

    # Generating Array of Hyperparameters
    lambda_min, lambda_max, N_lambda = 0.0001, 0.01, 50
    lambda_vals = np.linspace(lambda_min, lambda_max, N_lambda)

    # Creating Blank Arrays

    ridge_data["R2"] = np.zeros(N_lambda)
    ridge_data["MSE"] = np.zeros(N_lambda)
    ridge_data["var"] = np.zeros((N_lambda, degree**2 - degree + 1))

    tot = len(lambda_vals)

    for n,l in enumerate(lambda_vals):
        R.reset()
        R.poly(degree = degree, alpha = l)

        ridge_data["R2"][n], ridge_data["MSE"][n], ridge_data["var"][n] = \
        R.k_fold(k = k, degree = degree, sigma = sigma)

        print(f"\r{int(100*(n+1)/tot)}%", end = "")
    print("\r    ")

    plt.plot(lambda_vals, ridge_data["R2"])
    plt.xlabel("$\lambda$")
    plt.ylabel("$R^2$")
    plt.figure()
    plt.plot(lambda_vals, ridge_data["MSE"])
    plt.xlabel("$\lambda$")
    plt.ylabel("$MSE$")
    plt.show()

# part_a()
# part_b()
# part_d()
R.reset()
R.lasso(degree = 5, alpha = 0.1)
R.plot()
