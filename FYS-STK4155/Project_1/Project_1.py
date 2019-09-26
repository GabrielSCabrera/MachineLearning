from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from imageio import imread
from matplotlib import cm
import numpy as np
import sys, franke

sys.path.append("..")
from utils.classes import Regression

# Conditions
k = 5               # k in k-fold
degree = 5          # Polynomial approximation degree
sigma = 1           # Variance of Gaussian Noise
split_test = 20     # Percentage of data to split into testing set
alpha = 0.1

# Select random seed for consistent results
np.random.seed(69420666)

# Generating NxN meshgrid of x,y values in range [0, 1]
x_min, x_max, N = 0, 1, 100
x = np.linspace(x_min, x_max, int(N))
X,Y = np.meshgrid(x, x)

# Calculating the values of the Franke function at each (x,y) coordinate
Z = franke.FrankeFunction(X,Y)
init_error = np.random.normal(0, 0.1, Z.shape)
Z = Z + init_error

# Making compatible input arrays for Regression object
x = np.zeros((X.shape[0]*X.shape[1], 2))
x[:,0] = X.flatten()
x[:,1] = Y.flatten()
y = Z.flatten()

# TEMPORARY TESTING VALUES
# X = np.random.random(size = 10000)
# Y = np.random.random(size = 10000)
# Z = franke.FrankeFunction(X, Y) + np.random.normal(0, .01, size = X.shape[0])
# x = np.zeros((X.shape[0], 2))
# x[:,0] = X
# x[:,1] = Y

# Creating Regression object with x and y
R = Regression(x, y)

def part_a(R, savename=None):

    """PART A"""
    print("\n" + "-"*80 + "\nPART A\n" + "-"*80)
    
#    R.reset()
    # Implementing 5th degree polynomial regression in 2-D
    R.poly(degree = degree, alpha = alpha)
    R.plot(plot_points = True, savename=savename)

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
    var = " & ".join(list(f"{i:.3g}" for i in OLS_data["var"]))
    print(f"\nVar(beta) = \n{var}")
    print(f"\nMSE = {OLS_data['MSE']:.2g}")
    print(f"\nR² = {OLS_data['R2']:.2g}")
    print(f"\nσ² = {sigma2}")
#    R.reset()

def part_b(R, savename=None):


    """PART B"""
    print("\n" + "-"*80 + "\nPART B\n" + "-"*80)
    
#    R.reset()
    # Creating <dict> of values for OLS k-fold
    kfold_data = {}

    # Implementing the k-fold algorithm
    kfold_data["R2"], kfold_data["MSE"], kfold_data["var"] = \
    R.k_fold(k = k, degree = degree, sigma = sigma)
    
#    sigma2 = R.sigma()

    # Displaying Results
    var = " & ".join(list(f"{i:.3g}" for i in kfold_data["var"]))
    print(f"\nVar(beta) = \n{var}")
    print(f"\nMSE = {kfold_data['MSE']:.2g}")
    print(f"\nR² = {kfold_data['R2']:.2g}")
#    print(f"\nσ² = {sigma2}")
    
    return kfold_data
#    R.reset()
    
    
def part_c(R):
    
    """PART C"""
    print("\n" + "-"*80 + "\nPART C\n" + "-"*80)
    
#    R.reset()
    part_b(R)
    

    #implements the Cost function
    y_data = R.predict(R._X)
    exp_y = np.mean(y_data)
    
    f_data = R._Y - init_error.flatten()
    err = 0
    for fi,yi in zip(f_data, y_data):
        err += (fi - exp_y)**2 - (yi - exp_y)**2
        
    err /= len(f_data)
    err += R.sigma()
    
    print(f"\nE = {err}")
#    print(err)
    
    

def part_d(R, savename_R2=None, savename_MSE=None):

    """PART D"""
    print("\n" + "-"*80 + "\nPART D\n" + "-"*80)

    # Creating <dict> of values for ridge regression
    ridge_data = {"ridge":{}, "k_fold":{}}

    # Generating Array of Hyperparameters
    lambda_min, lambda_max, N_lambda = -4, 1, 1000
    lambda_vals = np.logspace(lambda_min, lambda_max, N_lambda)

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
    plt.xscale("log")
    if savename_R2!=None:
        plt.savefig(savename_R2)
    plt.figure()
    plt.plot(lambda_vals, ridge_data["MSE"])
    plt.xlabel("$\lambda$")
    plt.ylabel("$MSE$")
    plt.xscale("log")
    if savename_MSE!=None:
        plt.savefig(savename_MSE)
    plt.show()
    R.reset()

def part_e(R, savename_R2=None, savename_MSE=None):

    """PART E"""
    print("\n" + "-"*80 + "\nPART E\n" + "-"*80)


    R.reset()
    R.lasso(5, 0.001)

    """
    print(R._beta)
    print(np.max(R._beta))
    print(np.min(R._beta))
    print()

    for i in R._beta:
        print(i)

    R.plot()

    """
    # Creating <dict> of values for ridge regression
    ridge_data = {"ridge":{}, "k_fold":{}}

    # Generating Array of Hyperparameters
    lambda_min, lambda_max, N_lambda = -4, 1, 1000
    lambda_vals = np.logspace(lambda_min, lambda_max, N_lambda)
    print(lambda_vals)

    # Creating Blank Arrays

    ridge_data["R2"] = np.zeros(N_lambda)
    ridge_data["MSE"] = np.zeros(N_lambda)
    ridge_data["var"] = np.zeros((N_lambda, degree**2 - degree + 1))

    tot = len(lambda_vals)

    for n,l in enumerate(lambda_vals):
        R.reset()
        R.lasso(degree = degree, alpha = l)

        ridge_data["R2"][n], ridge_data["MSE"][n], ridge_data["var"][n] = \
        R.k_fold(k = k, degree = degree, sigma = sigma)

        print(f"\r{int(100*(n+1)/tot)}%", end = "")
    print("\r    ")

    plt.plot(lambda_vals, ridge_data["R2"])
    plt.xlabel("$\lambda$")
    plt.ylabel("$R^2$")
    plt.xscale("log")
    if savename_R2!=None:
        plt.savefig(savename_R2)
    plt.figure()
    plt.plot(lambda_vals, ridge_data["MSE"])
    plt.xlabel("$\lambda$")
    plt.ylabel("$MSE$")
    plt.xscale("log")
    if savename_MSE!=None:
        plt.savefig(savename_MSE)
    plt.show()

def part_f(savename=None):

    # Load the terrain
    terrain1 = imread("SRTM_data_Norway_1.tif")
    # Show the terrain
    plt.figure()
    plt.title("Terrain over Norway 1")
    plt.imshow(terrain1, cmap="gray")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

    # Load the terrain
    terrain2 = imread("SRTM_data_Norway_2.tif")
    # Show the terrain
    plt.figure()
    plt.title("Terrain over Norway 2")
    plt.imshow(terrain2, cmap="gray")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig("results/part_g_input.pdf")
    plt.show()

    return(terrain2[::25, ::25])

def part_g(savename=None):

    ter_data = part_f()

    x = np.arange(0, ter_data.shape[1], 1)
    y = np.arange(0, ter_data.shape[0], 1)

    X,Y = np.meshgrid(x, y)
    x = np.zeros((X.shape[0]*X.shape[1], 2))
    x[:,0] = X.flatten()
    x[:,1] = Y.flatten()
    y = ter_data.flatten()

    TER = Regression(x,y)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    fig.set_size_inches(8, 6)
    fig.tight_layout()
    surf = ax.plot_surface(X, Y, ter_data, cmap=cm.terrain, antialiased=False)
    plt.show()
    part_a(TER)
    part_b(TER)


#part_a(R, savename="results/part_a_reg.pdf")
#part_b(R)
part_c(R)
#part_d(R, savename_MSE="results/part_d_reg_MSE.pdf", savename_R2="results/part_d_reg_R2.pdf")
#part_e(R, savename_MSE="results/part_e_reg_MSE.pdf", savename_R2="results/part_e_reg_R2.pdf")
#part_f(R)


#part_g()
