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

k_fold = 12          # k in k-fold
max_deg = 5          # Maximum polynomial approximation degree
sigma = 1            # Variance of Gaussian noise in Franke function
split_test = 20      # Percentage of data to split into testing set
sigma = 0.01         # Standard deviation of Gaussian noise in Franke function

alpha_min_R = 1E-12  # Minimum Lambda in Ridge
alpha_max_R = 1E0    # Maximum lambda in Ridge
alpha_min_L = 1E-12  # Minimum Lambda in LASSO
alpha_max_L = 1E-4   # Maximum lambda in LASSO
N_alpha_R = 40       # Number of lambdas to check for with Ridge in Part d)
N_alpha_L = 40       # Number of lambdas to check for with LASSO in Part e)

save_dir = "output"  # Default directory in which to save output files
plots = True         # Whether to generate plots
save_all = True      # Whether to save all data in the save_dir directory
debug_mode = True    # Print status to terminal
extension = "png"    # Extension (filetype) of saved plots (do not include ".")
bv_plots = 4         # Number of bias-variance tradeoff plots in Part d)
#cmap = cm.twilight   # Colormap to use in 3-D surface plots
alpha_3D = 0.5       # Transparency of 3-D surface plots, range -> [0,1]

# Path of .tif file containing real terrain data
terrain_data = "SRTM_data_Norway_1.tif"

# Preparing the save path
save_dir = os.path.join(os.getcwd(), save_dir)

# Select random seed for consistent results
np.random.seed(69420666)

"""Helper Functions"""

def generate_Franke_data(x_min = 0, x_max = 1, N = 100, sigma = 0.01):

    # Generating NxN meshgrid of x,y values in range [0, 1]
    x_min, x_max, N = 0, 1, 100
    x = np.linspace(x_min, x_max, int(N))
    X,Y = np.meshgrid(x, x)

    # Calculating the values of the Franke function at each (x,y) coordinate
    Z = franke.FrankeFunction(X,Y)
    init_error = np.random.normal(0, sigma, Z.shape)
    Z = Z + init_error

    # Making compatible input arrays for Regression object
    x = np.zeros((X.shape[0]*X.shape[1], 2))
    x[:,0] = X.flatten()
    x[:,1] = Y.flatten()
    y = Z.flatten()

    return x, y, init_error

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
        plt.legend(["Variance", "$MSE$", "$R^2$"])
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

def part_C(R, init_error = 0, save = False, plots = False):
    bias = []
    var = []
    mse = []
    cost_train = []
    cost_test = []

    debug_title("C")
    
    degrees = np.arange(0, 20 + 1) #an array for each of the degrees we are testing
    Errs = np.zeros_like(degrees, dtype=float) #an array for the errors in the training sample
    test_Errs = np.zeros_like(degrees, dtype=float) #an array for the errors in the test sample
    mses = np.zeros_like(degrees, dtype=float)
    mses_test = np.zeros_like(degrees, dtype=float)    
    tot = len(degrees)

    for i in degrees:
        R.reset()
        R.split(20) #splits the data into training and testing data
        R.poly(degree = i, alpha = 0.1)

        #implements the Cost function for training data
        y_data = R.predict(R._X)
        exp_y = np.mean(y_data)

        f_data = R._Y - np.delete(init_error.flatten(), R._test_idx)
        err = np.mean((f_data - exp_y)**2 + (y_data - exp_y)**2) + R.sigma()

        Errs[i] = err
        mses[i] = np.mean((R._Y - y_data)**2)

        #implements the Cost function for test data
        y_data_test = R.predict(R._X_test)
        exp_y_test = np.mean(y_data_test)

        f_data_test = R._Y_test - init_error.flatten()[R._test_idx]
        err_test = np.mean((f_data_test - exp_y_test)**2 + (y_data_test - exp_y_test)**2) + R.sigma()

        test_Errs[i] = err_test
        mses_test[i] = np.mean((R._Y_test - y_data_test)**2)
        print(f"\r{int(100*(i+1)/tot)}%", end = "")
    print("\r    ")

    m_deg = 7
    d_vals = np.arange(1, m_deg + 1)

    for d in d_vals:
        if globals()["debug_mode"] is True:
            print(f"\r{np.round(100*d/m_deg):>3.0f}%", end = "")
        R.reset()
        arg_idx = R.split(test_size = split_test)
        R.poly(degree = d)

        Y_hat_train = R.predict()
        Y_hat_test = R.predict(X = R._X_test)

        mse_step = R.mse(split = True)
        bias_step = np.mean(Y_hat_test) - np.mean(R._Y_test)
        var_step = np.mean(Y_hat_test**2) - np.mean(Y_hat_test)**2
        
        cost_train_step = np.mean((R._Y - Y_hat_train)**2)
        cost_test_step = np.mean((R._Y_test - Y_hat_test)**2)

        cost_train.append(cost_train_step)
        cost_test.append(cost_test_step)

        mse.append(mse_step)
        bias.append(bias_step**2)
        var.append(var_step)

    print()
    

    if plots is True:
        plt.plot(d_vals, mse)
        plt.plot(d_vals, bias)
        plt.plot(d_vals, var)
        plt.xlabel("Polynomial Degree")
#        plt.xlim(1, max_deg)
        plt.legend(["$MSE$", "Bias", "Variance"])
        if save is False:
            plt.show()
        else:
            plt.savefig(f"{save_dir}/part_C_1.{extension}", dpi = 250)
            plt.close()

        plt.plot(d_vals, cost_train)
        plt.plot(d_vals, cost_test)
        plt.show()
        plt.plot(degrees, Errs)
        plt.plot(degrees, test_Errs)
        plt.xlabel("Polynomial Degree")
#        plt.xlim(1, max_deg)
        plt.legend([r"$C(\mathbf{X}_{train},\beta)$",
                    r"$C(\mathbf{X}_{test},\beta)$"])
        if save is False:
            plt.show()
        else:
            plt.savefig(f"{save_dir}/part_C_2.{extension}", dpi = 250)
            plt.close()

def part_D(R, save = False, plots = False):

    debug_title("D")

    d_vals = np.arange(1, max_deg + 1)
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

    x, y, init_error = generate_Franke_data(sigma = sigma)
    R_Franke = Regression(x, y)

    """Parts a – e"""

    print("\n\n\t\tFRANKE FUNCTION\n\n")
#
#    part_A(R = R_Franke, save = save_all, plots = plots)
#    part_B(R = R_Franke, save = save_all, plots = plots)
    part_C(R = R_Franke, plots = plots, init_error=init_error)
#    part_D(R = R_Franke, save = save_all, plots = plots)
#    part_E(R = R_Franke, save = save_all, plots = plots)

    """Parts f and g"""
#
#    print("\n\n\t\tREAL DATA\n\n")
#
#    # Creating a new directory to save the real data
#    save_dir = "real_output"
#    if not os.path.exists(save_dir):
#        os.mkdir(save_dir)
#
#    x, y = part_F(save = save_all, plots = plots)
#    R_real = Regression(x, y)
#
#    part_A(R = R_real, save = save_all, plots = plots)
#    part_B(R = R_real, save = save_all, plots = plots)
#    part_C(R = R_real, save = save_all, plots = plots)
#    part_D(R = R_real, save = save_all, plots = plots)
#    part_E(R = R_real, save = save_all, plots = plots)
