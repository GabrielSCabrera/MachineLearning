from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import ignore_warnings
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from imageio import imread
from matplotlib import cm
import sys, franke, os
import numpy as np
import matplotlib

sys.path.append("..")
from utils.classes import Regression

"""Preparing Global Settings"""

k_fold = 5          # k in k-fold
max_deg = 5         # Maximum polynomial approximation degree
sigma = 1           # Variance of Gaussian noise in Franke function
split_test = 20     # Percentage of data to split into testing set

alpha_min = 1E-10   # Minimum Lambda in LASSO and Ridge
alpha_max = 1E-1    # Maximum lambda in LASSO and Ridge
N_alpha = 10        # Number of lambdas to check for in Parts d), e)

save_dir = "output" # Default directory in which to save output files
plots = True        # Whether to generate plots
save_all = True     # Whether to save all data in the save_dir directory
debug_mode = True   # Print status to terminal
extension = "png"   # Extension (filetype) of saved plots (do not include ".")
bv_plots = 4        # Number of bias-variance tradeoff plots in Part d)
cmap = cm.plasma    # Colormap to use in 3-D surface plots
alpha_3D = 0.5      # Transparency of 3-D surface plots, range -> [0,1]

# Path of .tif file containing real terrain data
terrain_data = "SRTM_data_Norway_1.tif"

# Preparing the save path
save_dir = os.path.join(os.getcwd(), save_dir)

# Select random seed for consistent results
np.random.seed(69420666)

"""Helper Functions"""

def generate_Franke_data(x_min = 0, x_max = 1, N = 100):

    # Generating NxN meshgrid of x,y values in range [0, 1]
    x_min, x_max, N = 0, 1, 100
    x = np.linspace(x_min, x_max, int(N))
    X,Y = np.meshgrid(x, x)

    # Calculating the values of the Franke function at each (x,y) coordinate
    Z = franke.FrankeFunction(X,Y)
    # Normalizing the Franke function
    Z = (Z - np.mean(Z))/np.std(Z)
    init_error = np.random.normal(0, 0.1, Z.shape)
    Z = Z + init_error

    # Making compatible input arrays for Regression object
    x = np.zeros((X.shape[0]*X.shape[1], 2))
    x[:,0] = X.flatten()
    x[:,1] = Y.flatten()
    y = Z.flatten()

    return x, y

def debug_title(letter):
    if globals()["debug_mode"] is True:
        text = f"Part {letter}"
        char = "~"       # Must be a single character
        l = (80-len(text))//2
        print(f"{char*l}{text}{char*l}\n")

"""Main Project Code"""

def part_A(R, save = False, plots = False):
    var = []
    mse = []
    R2 = []

    d_vals = np.arange(1, max_deg + 1, 1)

    debug_title("A")

    for d in d_vals:
        if globals()["debug_mode"] is True:
            print(f"\r{np.round(100*d/max_deg):>3.0f}%", end = "")
        R.reset()
        R.poly(degree = d)
        var.append(np.mean(R.variance()))
        mse.append(R.mse())
        R2.append(R.r_squared())
    print()

    if plots is True:
        plt.plot(d_vals, var)
        plt.plot(d_vals, mse)
        plt.plot(d_vals, R2)
        plt.xlabel("Polynomial Degree")
        plt.xlim(1, max_deg)
        plt.legend(["Mean Variance", "$MSE$", "$R^2$"])
        if save is False:
            plt.show()
        else:
            plt.savefig(f"{save_dir}/part_A.{extension}", dpi = 250)
            plt.close()

    if save is True:
        with open(f"{save_dir}/part_A.txt", "w+") as outfile:
            for n, (i,j,k) in enumerate(zip(var, mse, R2)):
                temp = (f"{'_'*80}\nDegree {n+1:d}\n\nMSE = {j:g}\nR2 = {k:g}"
                        f"\nmean(var) = {i:g}\n")
                outfile.write(temp)

def part_B(R, save = False, plots = False):
    mse = []
    R2 = []

    debug_title("B")

    d_vals = np.arange(1, max_deg + 1, 1)
    for d in d_vals:
        if globals()["debug_mode"] is True:
            print(f"\r{np.round(100*d/max_deg):>3.0f}%", end = "")
        R.reset()
        R2_step, mse_step = R.k_fold(k = k_fold, degree = d)
        mse.append(mse_step)
        R2.append(R2_step)

    print()

    if plots is True:
        plt.plot(d_vals, mse)
        plt.plot(d_vals, R2)
        plt.xlabel("Polynomial Degree")
        plt.xlim(1, max_deg)
        plt.legend(["$MSE$", "$R^2$"])
        if save is False:
            plt.show()
        else:
            plt.savefig(f"{save_dir}/part_B.{extension}", dpi = 250)
            plt.close()

    if save is True:
        with open(f"{save_dir}/part_B.txt", "w+") as outfile:
            for n, (i,j) in enumerate(zip(mse, R2)):
                temp = f"{'_'*80}\nDegree {n+1:d}\n\nMSE = {i:g}\nR2 = {j:g}\n"
                outfile.write(temp)

def part_C(R, save = False, plots = False):
    bias = []
    var = []
    mse = []

    debug_title("C")

    d_vals = np.arange(1, max_deg + 1)

    for d in d_vals:
        if globals()["debug_mode"] is True:
            print(f"\r{np.round(100*d/max_deg):>3.0f}%", end = "")
        R.reset()
        R.split(test_size = split_test)
        R.poly(degree = d)
        Y_hat = R.predict(R._X_test)

        mse_step = R.mse(split = True)
        bias_step = np.mean(Y_hat) - np.mean(R._Y_test)
        var_step = np.mean(Y_hat**2) - np.mean(Y_hat)**2

        mse.append(mse_step)
        bias.append(bias_step**2)
        var.append(var_step)
    print()

    if plots is True:
        plt.plot(d_vals, mse)
        plt.plot(d_vals, bias)
        plt.plot(d_vals, var)
        plt.xlabel("Polynomial Degree")
        plt.xlim(1, max_deg)
        plt.legend(["$MSE$", "Bias", "Variance"])
        if save is False:
            plt.show()
        else:
            plt.savefig(f"{save_dir}/part_C.{extension}", dpi = 250)
            plt.close()

def part_D(R, save = False, plots = False):

    debug_title("D")

    d_vals = np.arange(1, max_deg + 1)
    lambda_vals = np.linspace(alpha_min, alpha_max, N_alpha)

    var = []
    mse = []
    R2 = []
    bias = []

    for m,l in enumerate(lambda_vals):
        var_step = []
        mse_step = []
        R2_step = []
        bias_step = []

        for n,d in enumerate(d_vals):
            if globals()["debug_mode"] is True:
                print(f"\r{(100*(n + m*len(d_vals) + 1)/(len(d_vals)*len(lambda_vals))):>3.0f}%", end = "")
            R.reset()
            R.split(test_size = split_test)
            R.poly(degree = d, alpha = l)
            Y_hat = R.predict()
            var_step.append(np.mean(R.variance(split = True)))
            mse_step.append(R.mse(split = True))
            R2_step.append(R.r_squared(split = True))
            bias_step.append(np.mean(Y_hat) - np.mean(R._Y))

        var.append(var_step)
        mse.append(mse_step)
        R2.append(R2_step)
        bias.append(bias_step)

    print()

    L,D = np.meshgrid(d_vals, lambda_vals)
    var, mse, R2, bias = \
    np.array(var), np.array(mse), np.array(R2), np.array(bias)**2

    if plots is True:
        data = [var, mse, R2, bias]
        data_labels = ["Variance", "$MSE$", "$R^2$", "Bias²"]
        xlabel = "Polynomial Degree"
        ylabel = r"Hyperparameter $\lambda$"

        for n,(i,j) in enumerate(zip(data, data_labels)):
            fig = plt.figure()
            ax = fig.gca(projection="3d")
            fig.set_size_inches(8, 6)

            ax.plot_surface(L, D, i, cmap = cmap, alpha = alpha_3D)
            ax.set_xlabel("\n\n\n" + xlabel, linespacing = 3)
            ax.set_ylabel("\n\n\n" + ylabel, linespacing = 3)
            ax.set_zlabel("\n\n\n" + j, linespacing = 3)

            if save is False:
                plt.show()
            else:
                plt.savefig(f"{save_dir}/part_D_{n+1:d}.{extension}",
                dpi = 250)
                plt.close()

        """Bias-Variance Curves"""

        s = len(lambda_vals)//globals()["bv_plots"]
        for n,(i,j,k) in enumerate(zip(bias[::s], var[::s], mse[::s])):
            plt.plot(d_vals, i, label = "Bias²")
            plt.plot(d_vals, j, label = "Variance")
            plt.plot(d_vals, k, label = "$MSE$")
            plt.legend()
            plt.xlabel(xlabel)
            plt.text(np.median(d_vals), 2*(np.max([i,j,k])-np.min([i,j,k]))/3,
             f"$\\lambda = {lambda_vals[s*n]:.2E}$")
            plt.xlim([1, max_deg])

            if save is False:
                plt.show()
            else:
                plt.savefig(f"{save_dir}/part_D_{len(data)+n+1:d}.{extension}",
                dpi = 250)
                plt.close()

@ignore_warnings(category = ConvergenceWarning)
def part_E(R, save = False, plots = False):

    debug_title("E")

    d_vals = np.arange(1, max_deg + 1)
    lambda_vals = np.linspace(alpha_min, alpha_max, N_alpha)

    var = []
    mse = []
    R2 = []
    bias = []

    for m,l in enumerate(lambda_vals):
        var_step = []
        mse_step = []
        R2_step = []
        bias_step = []

        for n,d in enumerate(d_vals):
            if globals()["debug_mode"] is True:
                print(f"\r{(100*(n + m*len(d_vals) + 1)/(len(d_vals)*len(lambda_vals))):>3.0f}%", end = "")
            R.reset()
            R.split(test_size = split_test)
            R.lasso(degree = d, alpha = l)
            Y_hat = R.predict(X = R._X_test)
            var_step.append(np.mean(Y_hat**2) - np.mean(Y_hat)**2)
            mse_step.append(R.mse())
            R2_step.append(R.r_squared())
            bias_step.append(np.mean(Y_hat) - np.mean(R._Y_test))

        var.append(var_step)
        mse.append(mse_step)
        R2.append(R2_step)
        bias.append(bias_step)

    print()

    L,D = np.meshgrid(d_vals, lambda_vals)
    var, mse, R2, bias = \
    np.array(var), np.array(mse), np.array(R2), np.array(bias)**2

    if plots is True:
        data = [var, mse, R2, bias]
        data_labels = ["Variance", "$MSE$", "$R^2$", "Bias²"]
        xlabel = "Polynomial Degree"
        ylabel = r"Hyperparameter $\lambda$"

        for n,(i,j) in enumerate(zip(data, data_labels)):
            fig = plt.figure()
            ax = fig.gca(projection="3d")
            fig.set_size_inches(8, 6)

            ax.plot_surface(L, D, i, cmap = cmap, alpha = alpha_3D)
            ax.set_xlabel("\n\n\n" + xlabel, linespacing = 3)
            ax.set_ylabel("\n\n\n" + ylabel, linespacing = 3)
            ax.set_zlabel("\n\n\n" + j, linespacing = 3)

            if save is False:
                plt.show()
            else:
                plt.savefig(f"{save_dir}/part_E_{n+1:d}.{extension}",
                dpi = 250)
                plt.close()

        """Bias-Variance Curves"""

        s = len(lambda_vals)//globals()["bv_plots"]
        for n,(i,j,k) in enumerate(zip(bias[::s], var[::s], mse[::s])):
            plt.plot(d_vals, i, label = "Bias²")
            plt.plot(d_vals, j, label = "Variance")
            plt.plot(d_vals, k, label = "$MSE$")
            plt.legend()
            plt.xlabel(xlabel)
            plt.text(np.median(d_vals), 2*(np.max([i,j,k])-np.min([i,j,k]))/3,
             f"$\\lambda = {lambda_vals[s*n]:.2E}$")
            plt.xlim([1, max_deg])

            if save is False:
                plt.show()
            else:
                plt.savefig(f"{save_dir}/part_E_{len(data)+n+1:d}.{extension}",
                dpi = 250)
                plt.close()

def part_F(save = False, plots = False):

    debug_title("F")

    if plots is True:
        # Importing the data
        terrain_data = imread(globals()["terrain_data"])
        # Resizing the data
        terrain_data = terrain_data[::10,::10]
        # Normalizing the data
        terrain_data = (terrain_data - np.mean(terrain_data))\
                       /np.std(terrain_data)
        figure = plt.imshow(terrain_data, cmap="gray")
        plt.xlabel("\n$x$")
        plt.ylabel("\n$y$")
        cbar = plt.colorbar(figure)
        cbar.ax.set_ylabel('\n\n$z$', rotation = 90)
        if save is False:
            plt.show()
        else:
            plt.savefig(f"{save_dir}/part_F.{extension}", dpi = 250)
            plt.close()

    print("100%")
    dims = terrain_data.shape
    x, y = np.arange(0, dims[0]), np.arange(0, dims[1])
    X, Y = np.meshgrid(x,y)
    X_regr = np.array([X.flatten(), Y.flatten()]).T
    Y_regr = terrain_data.flatten()

    return X_regr, Y_regr

if __name__ == "__main__":

    # Creating a directory to save the Franke function data
    save_dir = "franke_output"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    x, y = generate_Franke_data()
    R_Franke = Regression(x, y)

    """Parts a – e"""

    print("\n\n\t\tFRANKE FUNCTION\n\n")

    part_A(R = R_Franke, save = save_all, plots = plots)
    part_B(R = R_Franke, save = save_all, plots = plots)
    part_C(R = R_Franke, save = save_all, plots = plots)
    part_D(R = R_Franke, save = save_all, plots = plots)
    part_E(R = R_Franke, save = save_all, plots = plots)

    """Parts f and g"""

    print("\n\n\t\tREAL DATA\n\n")

    # Creating a new directory to save the real data
    save_dir = "real_output"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    x, y = part_F(save = save_all, plots = plots)
    R_real = Regression(x, y)

    part_A(R = R_real, save = save_all, plots = plots)
    part_B(R = R_real, save = save_all, plots = plots)
    part_C(R = R_real, save = save_all, plots = plots)
    part_D(R = R_real, save = save_all, plots = plots)
    part_E(R = R_real, save = save_all, plots = plots)
