import sys, franke
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations

sys.path.append("..")
# from utils.functions import least_squares_poly

np.random.seed(69420666)

def least_squares_poly(X, y, degree):
    """
        ---PURPOSE------------------------------------

        Finds the vector of coefficients for a polynomial of varying degree and
        a set of X and y points.

        ---INPUT--------------------------------------

        X           2-D Array of shape (N,p)
        y           1-D Array of shape (N,)
        degree      Integer greater than zero

        ---OUTPUT-------------------------------------

        beta        NumPy 1-D array
        exponents   NumPy 2-D array

    """

    X = np.array(X)
    y = np.array(y)

    if x.shape[0] != y.shape[0]:
        msg = "\n\nArguments <x> and <y> in function <least_squares_poly> must be of "
        msg += f"the same shape.  \n\n\tx.shape = {x.shape}\ny.shape = {y.shape}"
        raise Exception(msg)

    if len(y.shape) != 1:
        msg = "\n\nArgument <y> in function <least_squares_poly> must be "
        msg += f"one dimensional.  \n\n\ty.shape = {y.shape}"
        raise Exception(msg)

    N = X.shape[0]
    p = X.shape[1]

    try:
        if degree == int(degree) and degree > 0:
            degree = int(degree)
        else:
            msg = "\n\nArgument <degree> in function <least_squares_poly> must be an "
            msg += f"integer greater than zero.  \ndegree = {degree}"
            raise Exception(msg)
    except ValueError:
        msg = "\n\nArgument <degree> in function <least_squares_poly> must be a "
        msg += f"number.  \n\n\ttype(degree) = {type(degree)}"
        raise Exception(msg)

    M = int(degree) + 1

    powers = np.arange(0, M, 1)
    exponents = list(permutations(powers, 2))

    for power in powers:
        exponents.append(power*np.ones(p))

    expo_sum = np.sum(exponents, axis = 1)
    valid_idx = np.where(np.less_equal(expo_sum, degree))[0]

    exponents = np.array(exponents)
    exponents = exponents[valid_idx]

    A = np.zeros((N, exponents.shape[0]))

    for n,exponent in enumerate(exponents):
        A[:,n] = np.prod(X**exponent, axis = 1)

    beta = (np.linalg.inv(A.T @ A) @ A.T) @ y
    return beta, exponents

x = np.linspace(0.1,0.9,25)
X,Y = np.meshgrid(x, x)

Z = franke.FrankeFunction(X,Y)
Z = Z + np.random.normal(0, 0.1, Z.shape)

x = np.zeros((X.shape[0]*X.shape[1], 2))

x[:,0] = X.flatten()
x[:,1] = Y.flatten()
y = Z.flatten()

fig = plt.figure()
ax = fig.gca(projection="3d")
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

beta, exponents = least_squares_poly(x, y, 5)

Z2 = np.zeros_like(Z)
for b,e in zip(beta, exponents):
    Z2 = Z2 + b*(X**e[0])*(Y**e[1])

fig = plt.figure()
ax = fig.gca(projection="3d")
surf = ax.plot_surface(X, Y, Z2, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
# ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
