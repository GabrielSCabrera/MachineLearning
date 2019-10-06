from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import ignore_warnings
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from imageio import imread
from matplotlib import cm
import sys, franke, os
from time import time
import numpy as np
import matplotlib

sys.path.append("..")
from utils.classes import Regression

"""Preparing Global Settings"""

k_fold = 12          # k in k-fold
min_deg = 1          # Minimum polynomial approximation degree
max_deg = 10         # Maximum polynomial approximation degree
split_test = 25      # Percentage of data to split into testing set
sigma = 0.01         # Standard deviation of Gaussian noise in Franke function

alpha_min_R = 1E-16  # Minimum Lambda in Ridge
alpha_max_R = 1E0    # Maximum lambda in Ridge
alpha_min_L = 1E-16  # Minimum lambda in LASSO
alpha_max_L = 1E-3   # Maximum lambda in LASSO
N_alpha_R = 30       # Number of lambdas to check for with Ridge in Part d)
N_alpha_L = 30       # Number of lambdas to check for with LASSO in Part e)

save_dir = "output"  # Default directory in which to save output files
plots = True         # Whether to generate plots
save_all = True      # Whether to save all data in the save_dir directory
debug_mode = True    # Print status to terminal
extension = "png"    # Extension (filetype) of saved plots (do not include ".")
bv_plots = 4         # Number of bias-variance tradeoff plots in Part d)
cmap = cm.magma      # Colormap to use in 3-D surface plots
alpha_3D = 0.5       # Transparency of 3-D surface plots, range -> [0,1]

# Path of .tif file containing real terrain data
terrain_data = "SRTM_data_Norway_2.tif"

# Generating array of dimensions
d_vals = np.arange(min_deg, max_deg + 1, 1)

# Preparing the save path
save_dir = os.path.join(os.getcwd(), save_dir)

# Select random seed for consistent results
np.random.seed(69420666)

"""Helper Functions"""

def plot_Franke(x_min = 0, x_max = 1, N = 100):
    # Generating NxN meshgrid of x,y values in range [0, 1]
    x_min, x_max, N = 0, 1, 100
    x = np.linspace(x_min, x_max, int(N))
    X,Y = np.meshgrid(x, x)

    # Calculating the values of the Franke function at each (x,y) coordinate
    Z = franke.FrankeFunction(X,Y)
    init_error = np.random.normal(0, globals()["sigma"], Z.shape)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    fig.set_size_inches(8, 6)
    fig.tight_layout()

    ax.plot_surface(X, Y, Z, cmap = cmap, alpha = alpha_3D)
    ax.set_xlabel("\n$x$", linespacing = 3)
    ax.set_ylabel("\n$y$", linespacing = 3)
    ax.set_zlabel("\n$f(x,y)$", linespacing = 3)

    plt.savefig(f"{save_dir}/Franke.{extension}",  dpi = 250)
    plt.close()

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    fig.set_size_inches(8, 6)
    fig.tight_layout()

    ax.plot_surface(X, Y, Z+init_error, cmap = cmap, alpha = alpha_3D)
    ax.set_xlabel("\n$x$", linespacing = 3)
    ax.set_ylabel("\n$y$", linespacing = 3)
    ax.set_zlabel("\n$f(x,y)$", linespacing = 3)

    plt.savefig(f"{save_dir}/Franke_noise.{extension}",  dpi = 250)
    plt.close()

def generate_Franke_data(x_min = 0, x_max = 1, N = 150):

    # Generating NxN meshgrid of x,y values in range [x_min, x_max]
    X = np.random.random((N,N))*(x_max-x_min) + x_min
    Y = np.random.random((N,N))*(x_max-x_min) + x_min

    # Calculating the values of the Franke function at each (x,y) coordinate
    Z = franke.FrankeFunction(X,Y)
    init_error = np.random.normal(0, globals()["sigma"], Z.shape)

    # Normalizing Z
    # Z = (Z - np.mean(Z)) /np.std(Z)
    # Z = (Z - np.min(Z))/(np.max(Z)-np.min(Z))
    f_xy = Z.copy().flatten()
    Z = Z + init_error

    # Making compatible input arrays for Regression object
    x = np.zeros((X.shape[0]*X.shape[1], 2))
    x[:,0] = X.flatten()
    x[:,1] = Y.flatten()
    y = Z.flatten()

    return x, y, f_xy

def debug_title(letter):
    if globals()["debug_mode"] is True:
        text = f"Part {letter}"
        char = "~"       # Must be a single character
        l = (80-len(text))//2
        print(f"{char*l}{text}{char*l}\n")

"""Main Project Code"""

def part_A(R, save = False, plots = False, name = ""):
    var = []
    mse = []
    R2 = []

    debug_title("A")

    for d in d_vals:
        if globals()["debug_mode"] is True:
            print(f"\r{np.round(100*d/max_deg):>3.0f}%", end = "")
        R.reset()
        R.poly(degree = d)
        Y_hat = R.predict()
        var.append(np.mean((Y_hat - np.mean(Y_hat))**2))
        mse.append(np.mean((R._Y - Y_hat)**2))
        R2.append(R.r_squared())
    print()

    if plots is True:
        plt.plot(d_vals, var)
        plt.plot(d_vals, mse)
        plt.plot(d_vals, R2)
        plt.xlabel("Polynomial Degree")
        plt.xlim(min_deg, max_deg)
        plt.legend(["Variance", "$MSE$", "$R^2$"])
        if save is False:
            plt.show()
        else:
            plt.savefig(f"{save_dir}/part_A{name}.{extension}", dpi = 250)
            plt.close()

    if save is True:
        with open(f"{save_dir}/part_A.txt", "w+") as outfile:
            for n, (i,j,k) in enumerate(zip(var, mse, R2)):
                temp = (f"{'_'*80}\nDegree {n+1:d}\n\nMSE = {j:g}\nR2 = {k:g}"
                        f"\nmean(var) = {i:g}\n")
                outfile.write(temp)

def part_B(R, save = False, plots = False, name = ""):
    mse = []
    R2 = []
    var = []

    debug_title("B")

    for d in d_vals:
        if globals()["debug_mode"] is True:
            print(f"\r{np.round(100*d/max_deg):>3.0f}%", end = "")
        R.reset()
        R2_step, mse_step, var_step = R.k_fold(k = k_fold, degree = d)
        mse.append(mse_step)
        R2.append(R2_step)
        var.append(var_step)

    print()

    if plots is True:
        plt.plot(d_vals, var)
        plt.plot(d_vals, mse)
        plt.plot(d_vals, R2)
        plt.xlabel("Polynomial Degree")
        plt.xlim(min_deg, max_deg)
        plt.legend(["Variance", "$MSE$", "$R^2$"])
        if save is False:
            plt.show()
        else:
            plt.savefig(f"{save_dir}/part_B{name}.{extension}", dpi = 250)
            plt.close()

    if save is True:
        with open(f"{save_dir}/part_B.txt", "w+") as outfile:
            for n, (i,j) in enumerate(zip(mse, R2)):
                temp = f"{'_'*80}\nDegree {n+1:d}\n\nMSE = {i:g}\nR2 = {j:g}\n"
                outfile.write(temp)

def part_C(R, f_xy = None, save = False, plots = False, name = ""):
    bias = []
    var = []
    mse = []

    cost_train = []
    cost_test = []

    debug_title("C")

    for n,d in enumerate(d_vals):
        if globals()["debug_mode"] is True:
            print(f"\r{np.round(100*(n+1)/len(d_vals)):>3.0f}%", end = "")
        R.reset()

        if f_xy is not None:
            R2_step, mse_step, var_step, bias_step = \
            R.k_fold(k = k_fold, degree = d, f_xy = f_xy)
            bias.append(bias_step)
            var.append(var_step)
        else:
            R2_step, mse_step, var_step = R.k_fold(k = k_fold, degree = d)

        R.reset()
        arg_idx = R.split(test_size = split_test)
        R.poly(degree = d)
        Y_hat_train = R.predict()
        Y_hat_test = R.predict(X = R._X_test)

        # if f_xy is not None:
        #     f_train = f_xy[arg_idx[0]]
        #     f_test = f_xy[arg_idx[1]]
        #
        #     bias_step = np.mean((f_test - np.mean(Y_hat_test))**2)
        #     var_step = np.mean((Y_hat_test - np.mean(Y_hat_test))**2)

        cost_train_step = np.mean((R._Y - Y_hat_train)**2)
        cost_test_step = np.mean((R._Y_test - Y_hat_test)**2)

        cost_train.append(cost_train_step)
        cost_test.append(cost_test_step)

    print()

    if plots is True:
        if f_xy is not None:
            plt.plot(d_vals, bias)
            plt.plot(d_vals, var)
            plt.xlabel("Polynomial Degree")
            plt.xlim(min_deg, max_deg)
            plt.legend(["Bias", "Variance"])
            if save is False:
                plt.show()
            else:
                plt.savefig(f"{save_dir}/part_C_1{name}.{extension}", dpi = 250)
                plt.close()

        plt.plot(d_vals, cost_train)
        plt.plot(d_vals, cost_test)
        plt.xlabel("Polynomial Degree")
        plt.xlim(min_deg, max_deg)
        plt.legend([r"$MSE(\mathbf{X}_{train})$", r"$MSE(\mathbf{X}_{test})$"])
        if save is False:
            plt.show()
        else:
            plt.savefig(f"{save_dir}/part_C_2{name}.{extension}", dpi = 250)
            plt.close()

def part_D(R, f_xy = None, save = False, plots = False, name = ""):

    debug_title("D")

    lambda_vals = np.linspace(alpha_min_R, alpha_max_R, N_alpha_R)

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
            arg_idx = R.split(test_size = split_test)

            if f_xy is not None:
                R2_step_2, mse_step_2, var_step_2, bias_step_2 = \
                R.k_fold(k = k_fold, degree = d, f_xy = f_xy, alpha = l)
                bias_step.append(bias_step_2)
                var_step.append(var_step_2)
                mse_step.append(mse_step_2)
                R2_step.append(R2_step_2)
            else:
                R2_step_2, mse_step_2, var_step_2 = \
                R.k_fold(k = k_fold, degree = d, alpha = l)
                var_step.append(var_step_2)
                mse_step.append(mse_step_2)
                R2_step.append(R2_step_2)

            if f_xy is None:
                bias_step.append(0)

        var.append(var_step)
        mse.append(mse_step)
        R2.append(R2_step)
        bias.append(bias_step)

    print()

    D,L = np.meshgrid(d_vals, lambda_vals)
    var, mse, R2, bias = \
    np.array(var), np.array(mse), np.array(R2), np.array(bias)

    if plots is True:

        if f_xy is None:
            data = [var, mse, R2]
            data_labels = ["Variance", "$MSE$", "$R^2$"]
        else:
            data = [var, mse, R2, bias]
            data_labels = ["Variance", "$MSE$", "$R^2$", "Bias²"]

        xlabel = "Polynomial Degree"
        ylabel = r"Hyperparameter $\lambda$"

        for n,(i,j) in enumerate(zip(data, data_labels)):
            fig = plt.figure()
            ax = fig.gca(projection="3d")
            fig.set_size_inches(8, 6)

            if j in ["$MSE$", "Bias²","Variance"]:
                minmax = "Minimum"
                idx = np.unravel_index(i.argmin(), i.shape)
            else:
                minmax = "Maximum"
                idx = np.unravel_index(i.argmax(), i.shape)

            ax.plot([D[idx]],[L[idx]],[i[idx]], marker = "X",
            markersize = 10, markeredgecolor = "k", markerfacecolor = "g")

            legend = (f"{minmax} {j} = {i[idx]:g} at\n$d$ = {D[idx]:g}, $\\lambda$ = "
                      f"{L[idx]:g}")

            plt.legend([legend])

            ax.plot_surface(D, L, i, cmap = cmap, alpha = alpha_3D)
            ax.set_xlabel("\n\n\n" + xlabel, linespacing = 3)
            ax.set_ylabel("\n\n\n" + ylabel, linespacing = 3)
            ax.set_zlabel("\n\n\n" + j, linespacing = 3)

            if save is False:
                plt.show()
            else:
                plt.savefig(f"{save_dir}/part_D_{n+1:d}{name}.{extension}",
                dpi = 250)
                plt.close()

        """Bias-Variance Curves"""

        if f_xy is not None:
            s = len(lambda_vals)//globals()["bv_plots"]
            for n,(i,j) in enumerate(zip(bias[::s], var[::s])):
                plt.plot(d_vals, i, label = "Bias²")
                plt.plot(d_vals, j, label = "Variance")
                plt.legend()
                plt.xlabel(xlabel)
                plt.text((np.max(d_vals)-np.min(d_vals))/2 + np.min(d_vals),
                         2*(np.max([i,j])-np.min([i,j]))/3,
                         f"$\\lambda = {lambda_vals[s*n]:.2E}$")
                plt.xlim([min_deg, max_deg])

                if save is False:
                    plt.show()
                else:
                    plt.savefig(f"{save_dir}/part_D_{len(data)+n+1:d}{name}.{extension}",
                    dpi = 250)
                    plt.close()

@ignore_warnings(category = ConvergenceWarning)
def part_E(R, f_xy = None, save = False, plots = False, name = ""):

    debug_title("E")

    lambda_vals = np.linspace(alpha_min_L, alpha_max_L, N_alpha_L)

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
            arg_idx = R.split(test_size = split_test)

            if f_xy is not None:
                R2_step_2, mse_step_2, var_step_2, bias_step_2 = \
                R.k_fold_lasso(k = k_fold, degree = d, f_xy = f_xy, alpha = l)
                bias_step.append(bias_step_2)
                var_step.append(var_step_2)
                mse_step.append(mse_step_2)
                R2_step.append(R2_step_2)
            else:
                R2_step_2, mse_step_2, var_step_2 = \
                R.k_fold_lasso(k = k_fold, degree = d, alpha = l)
                var_step.append(var_step_2)
                mse_step.append(mse_step_2)
                R2_step.append(R2_step_2)

            if f_xy is None:
                bias_step.append(0)

        var.append(var_step)
        mse.append(mse_step)
        R2.append(R2_step)
        bias.append(bias_step)

    print()

    D,L = np.meshgrid(d_vals, lambda_vals)
    var, mse, R2, bias = \
    np.array(var), np.array(mse), np.array(R2), np.array(bias)

    if plots is True:

        if f_xy is None:
            data = [var, mse, R2]
            data_labels = ["Variance", "$MSE$", "$R^2$"]
        else:
            data = [var, mse, R2, bias]
            data_labels = ["Variance", "$MSE$", "$R^2$", "Bias²"]

        xlabel = "Polynomial Degree"
        ylabel = r"Hyperparameter $\lambda$"

        for n,(i,j) in enumerate(zip(data, data_labels)):
            fig = plt.figure()
            ax = fig.gca(projection="3d")
            fig.set_size_inches(8, 6)

            if j in ["$MSE$", "Bias²", "Variance"]:
                minmax = "Minimum"
                idx = np.unravel_index(i.argmin(), i.shape)
            else:
                minmax = "Maximum"
                idx = np.unravel_index(i.argmax(), i.shape)
            ax.plot([D[idx]],[L[idx]],[i[idx]], marker = "X",
            markersize = 10, markeredgecolor = "k", markerfacecolor = "g")

            legend = (f"{minmax} {j} = {i[idx]:g} at\n$d$ = {D[idx]:g}, $\\lambda$ = "
                      f"{L[idx]:g}")

            plt.legend([legend])

            ax.plot_surface(D, L, i, cmap = cmap, alpha = alpha_3D)
            ax.set_xlabel("\n\n\n" + xlabel, linespacing = 3)
            ax.set_ylabel("\n\n\n" + ylabel, linespacing = 3)
            ax.set_zlabel("\n\n\n" + j, linespacing = 3)

            if save is False:
                plt.show()
            else:
                plt.savefig(f"{save_dir}/part_E_{n+1:d}{name}.{extension}",
                dpi = 250)
                plt.close()

        """Bias-Variance Curves"""

        if f_xy is not None:
            s = len(lambda_vals)//globals()["bv_plots"]
            for n,(i,j) in enumerate(zip(bias[::s], var[::s])):
                plt.plot(d_vals, i, label = "Bias²")
                plt.plot(d_vals, j, label = "Variance")
                plt.legend()
                plt.xlabel(xlabel)
                plt.text((np.max(d_vals)-np.min(d_vals))/2 + np.min(d_vals),
                         2*(np.max([i,j])-np.min([i,j]))/3,
                         f"$\\lambda = {lambda_vals[s*n]:.2E}$")
                plt.xlim([min_deg, max_deg])

                if save is False:
                    plt.show()
                else:
                    plt.savefig(f"{save_dir}/part_E_{len(data)+n+1:d}{name}.{extension}",
                    dpi = 250)
                    plt.close()

def part_F(save = False, plots = False):

    debug_title("F")

    if plots is True:
        # Importing the data
        terrain_data = imread(globals()["terrain_data"])
        # Resizing the data
        terrain_data = terrain_data[::15,::15]
        # Normalizing the data
        terrain_data = (terrain_data - np.mean(terrain_data))\
                       / np.std(terrain_data)
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

    t1 = time()

    # Creating a directory to save the Franke function data
    save_dir = "franke_output"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if plots is True:
        plot_Franke(x_min = 0, x_max = 1, N = 100)

    x, y, f_xy = generate_Franke_data()
    R_Franke = Regression(x, y)

    """Parts a – e"""

    print("\n\n\t\tFRANKE FUNCTION\n\n")

    part_A(R = R_Franke, save = save_all, plots = plots)
    part_B(R = R_Franke, save = save_all, plots = plots)
    part_C(R = R_Franke, f_xy = f_xy, save = save_all, plots = plots)
    part_D(R = R_Franke, f_xy = f_xy, save = save_all, plots = plots)
    part_E(R = R_Franke, f_xy = f_xy, save = save_all, plots = plots)

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

    dt = time() - t1
    print(f"Elapsed time: {dt:} seconds")
