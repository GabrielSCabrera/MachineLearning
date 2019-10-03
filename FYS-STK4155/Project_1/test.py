from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import franke, sys
import numpy as np

sys.path.append("..")
from utils.classes import Regression

sigma = 0.01

def generate_Franke_data(x_min = 0, x_max = 1, N = 100, sigma = 0.01):

    # Generating NxN meshgrid of x,y values in range [0, 1]
    x_min, x_max, N = 0, 1, 300
    x = np.linspace(x_min, x_max, int(N))
    X,Y = np.meshgrid(x, x)

    # Calculating the values of the Franke function at each (x,y) coordinate
    Z = franke.FrankeFunction(X,Y)
    init_error = np.random.normal(0, globals()["sigma"], Z.shape)
    Z = Z + init_error

    # Normalizing Z
    Z = (Z - np.mean(Z))/np.std(Z)

    # Making compatible input arrays for Regression object
    x = np.zeros((X.shape[0]*X.shape[1], 2))
    x[:,0] = X.flatten()
    x[:,1] = Y.flatten()
    y = Z.flatten()

    return x, y

if __name__ == "__main__":

    x,y = generate_Franke_data()
    R = Regression(x,y)
    mse = []
    mse2 = []
    degs = np.arange(2,8)
    for d in degs:
        R.reset()
        R.split(20)
        R.poly(degree = d)
        mse.append(R.mse(split = True))
        mse2.append(R.mse(split = False))

    plt.plot(degs, mse)
    plt.figure()
    plt.plot(degs, mse2)
    plt.show()

    # fig = plt.figure()
    # ax = fig.gca(projection="3d")
    # fig.set_size_inches(8, 6)
    #
    # ax.plot_surface(L, D, i, cmap = cmap, alpha = alpha_3D)
    # ax.set_xlabel("\n\n\n" + xlabel, linespacing = 3)
    # ax.set_ylabel("\n\n\n" + ylabel, linespacing = 3)
    # ax.set_zlabel("\n\n\n" + j, linespacing = 3)
    #
    # if save is False:
    #     plt.show()
    # else:
    #     plt.savefig(f"{save_dir}/part_D_{n+1:d}.{extension}",
    #     dpi = 250)
    #     plt.close()
