from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from imageio import imread
from matplotlib import cm
import sys, franke, os
import numpy as np
import matplotlib

sys.path.append("..")
from utils.classes import Regression

# Conditions
k_fold = 5              # k in k-fold
max_deg = 5             # Maximum polynomial approximation degree
sigma = 1               # Variance of Gaussian noise in Franke function
split_test = 20         # Percentage of data to split into testing set

alpha = 1E-10           # Lambda in LASSO and minimum lambda in Ridge
alpha_max = 1E-1        # Maximum lambda in Ridge
N_alpha = 10            # Number of lambdas to check for in Part D)

save_dir = "output"     # Directory in which to save output files
plots = True            # Whether to generate plots
save_all = True         # Whether to save all data in the save_dir directory
debug_mode = True       # Print status to terminal

# Preparing the save path
save_dir = os.path.join(os.getcwd(), save_dir)

# Select random seed for consistent results
np.random.seed(69420666)

def generate_Franke_data(x_min = 0, x_max = 1, N = 100):

    # Generating NxN meshgrid of x,y values in range [0, 1]
    x_min, x_max, N = 0, 1, 100
    x = np.linspace(x_min, x_max, int(N))
    X,Y = np.meshgrid(x, x)

    # Calculating the values of the Franke function at each (x,y) coordinate
    Z = franke.FrankeFunction(X,Y)
    f_xy = Z.flatten()
    init_error = np.random.normal(0, 0.1, Z.shape)
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

def part_A(R, save = None, plots = False):
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
            plt.savefig(f"{save_dir}/part_A.png")
            plt.close()

    if save is True:
        with open(f"{save_dir}/part_A.txt", "w+") as outfile:
            for n, (i,j,k) in enumerate(zip(var, mse, R2)):
                temp = (f"{'_'*80}\nDegree {n+1:d}\n\nMSE = {j:g}\nR2 = {k:g}"
                        f"\nmean(var) = {i:g}\n")
                outfile.write(temp)

def part_B(R, save = None):
    mse = []
    R2 = []

    debug_title("B")

    for d in range(1, max_deg + 1):
        if globals()["debug_mode"] is True:
            print(f"\r{np.round(100*d/max_deg):>3.0f}%", end = "")
        R.reset()
        R2_step, mse_step = R.k_fold(k = k_fold, degree = d)
        mse.append(mse_step)
        R2.append(R2_step)

    print()

    if save is True:
        with open(f"{save_dir}/part_B.txt", "w+") as outfile:
            for n, (i,j) in enumerate(zip(mse, R2)):
                temp = f"{'_'*80}\nDegree {n+1:d}\n\nMSE = {i:g}\nR2 = {j:g}\n"
                outfile.write(temp)

def part_C(R, f_xy, save = None, plots = False):
    bias = []
    var = []
    err = []

    debug_title("C")

    d_vals = np.arange(1, max_deg + 1)

    for d in d_vals:
        if globals()["debug_mode"] is True:
            print(f"\r{np.round(100*d/max_deg):>3.0f}%", end = "")
        R.reset()
        R.split(test_size = split_test)
        R.poly(degree = d)
        Y_hat = R.predict(R._X_test)

        err_step = np.mean((R._Y_test - Y_hat)**2)
        bias_step = np.mean(Y_hat) - np.mean(R._Y_test)
        var_step = np.mean(Y_hat**2) - np.mean(Y_hat)**2

        err.append(err_step)
        bias.append(bias_step**2)
        var.append(var_step)
    print()

    if plots is True:
        plt.plot(d_vals, err)
        plt.plot(d_vals, bias)
        plt.plot(d_vals, var)
        plt.xlabel("Polynomial Degree")
        plt.xlim(1, max_deg)
        plt.legend(["Expected Error", "Bias", "Variance"])
        if save is False:
            plt.show()
        else:
            plt.savefig(f"{save_dir}/part_C.png")
            plt.close()

def part_D(R, save = None, plots = False):

    debug_title("D")

    d_vals = np.arange(1, max_deg + 1)
    lambda_vals = np.linspace(alpha, alpha_max, N_alpha)

    var = []
    mse = []
    R2 = []

    for m,l in enumerate(lambda_vals):
        var_step = []
        mse_step = []
        R2_step = []

        for n,d in enumerate(d_vals):
            if globals()["debug_mode"] is True:
                print(f"\r{(100*(n + m*len(d_vals) + 1)/(len(d_vals)*len(lambda_vals))):>3.0f}%", end = "")
            R.reset()
            R.poly(degree = d, alpha = l)
            var_step.append(np.mean(R.variance()))
            mse_step.append(R.mse())
            R2_step.append(R.r_squared())

        var.append(var_step)
        mse.append(mse_step)
        R2.append(R2_step)
    print()

    L,D = np.meshgrid(d_vals, lambda_vals)
    var, mse, R2 = np.array(var), np.array(mse), np.array(R2)

    if plots is True:
        data = [var, mse, R2]
        data_labels = ["Variance", "$MSE$", "$R^2$"]
        xlabel = "Polynomial Degree"
        ylabel = r"Hyperparameter $\lambda$"

        for n,(i,j) in enumerate(zip(data, data_labels)):
            fig = plt.figure()
            ax = fig.gca(projection="3d")
            fig.set_size_inches(8, 6)

            ax.plot_surface(L, D, i)
            ax.set_xlabel("\n\n\n" + xlabel, linespacing = 3)
            ax.set_ylabel("\n\n\n" + ylabel, linespacing = 3)
            ax.set_zlabel("\n\n\n" + j, linespacing = 3)

            if save is False:
                plt.show()
            else:
                plt.savefig(f"{save_dir}/part_D_{n+1:d}.png", dpi = 250)
                plt.close()

if __name__ == "__main__":
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    x, y, f_xy = generate_Franke_data()
    R_Franke = Regression(x, y)

    part_A(R = R_Franke, save = save_all, plots = plots)
    part_B(R = R_Franke, save = save_all)
    part_C(R = R_Franke, f_xy = f_xy, save = save_all, plots = plots)
    part_D(R = R_Franke, save = save_all, plots = plots)
