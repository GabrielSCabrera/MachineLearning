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

    beta = np.linalg.inv(A.T @ A) @ A.T @ y
    return beta, exponents

def least_squares_functions(X, y, fxn_list, variables):
    """
        ---PURPOSE------------------------------------

        Finds the vector of coefficients for a given set of functions and
        set of x and y points. The intercept/bias is automatically included as
        the first element of the returned <beta> array.

        ---INPUT--------------------------------------

        X           2-D Array of shape (N,p)
        y           Array of shape (N,)
        fxnlist     List of M strings
        variables   List of p strings

        ---OUTPUT-------------------------------------

        beta        NumPy array of shape (M+1,)

        ---NOTES--------------------------------------

        The strings in parameter <fxnlist> should be composed of mathematically
        evaluatable expressions.

        The strings in parameter <variables> should be single characters, each
        representing a variable used in the input functions.  Each string must
        be unique, and will represent one column in the X matrix.

        For example, to perform linear regression,
        we would pass the list ["x"]; for a quadratic equation, we would pass
        the list ["x", "x**2"].

        To match a trigonometric function of sines and cosines, we could
        pass ["np.sin(x)", "np.cos(x)"].

        (Tip: Make sure NumPy is imported in the above case!)
    """
    X = np.array(X)
    Y = np.array(y)

    if x.shape[0] != Y.shape[0]:
        msg = "\n\nArguments <x> and <y> in function <least_squares_functions> must be of "
        msg += f"the same shape.  \n\n\tx.shape = {x.shape}\ny.shape = {y.shape}"
        raise Exception(msg)

    if len(Y.shape) != 1:
        msg = "\n\nArgument <y> in function <least_squares_functions> must be "
        msg += f"one dimensional.  \n\n\ty.shape = {y.shape}"
        raise Exception(msg)

    if not isinstance(fxn_list, (list, tuple)):
        msg = "\n\nArgument <fxn_list> in function <least_squares_functions> must be a "
        msg += f"list or tuple.  \n\n\ttype(fxn_list) = {type(fxn_list)}"
        raise Exception(msg)

    if not isinstance(variables, (list, tuple)):
        msg = "\n\nArgument <variables> in function <least_squares_functions> must be a "
        msg += f"list or tuple.  \n\n\ttype(variables) = {type(variables)}"
        raise Exception(msg)

    local_scope = locals()
    for n,var in enumerate(variables):
        msg = "\n\nElements of <variables> in function <least_squares_functions> "
        msg += " must be unique, lowercase, single-character strings."
        condition1 = not isinstance(var, str)
        condition2 = len(var) != 1
        condition3 = variables.count(var) != 1
        condition4 = var != var.lower()

        if condition1 or condition2 or condition3 or condition4:
            raise Exception(msg)

        exec(f"{var} = X[:,{n}]", globals(), local_scope)

    for n,fxn in enumerate(fxn_list):
        if not isinstance(fxn, str):
            msg = "\n\nElements of <fxn_list> in function <least_squares_functions> must"
            msg += f" be strings.  \n\n\ttype(fxn_list[{n}]) = {type(fxn_list[n])}"
            raise Exception(msg)

    if "1" not in fxn_list:
        fxn_list = ["1"] + list(fxn_list)

    M = len(fxn_list)

    N = X.shape[0]
    p = X.shape[1]

    A = np.zeros((N, M))

    msg = "\n\nElements of <fxn_list> in function <least_squares_functions> "
    msg += " must be mathematically evaluatable."
    msg += f"\n\n\tfxn_list[{n}] = \"{fxn}\" is an invalid expression."
    msg += f"\n\nTroubleshooting:\n\tIs {fxn} in the standard "
    msg += "library?  If not, be sure to import it!"

    for n, fxn in enumerate(fxn_list):
        try:
            value = eval(fxn, globals(), local_scope)
        except NameError:
            raise Exception(msg)

        if isinstance(value, (float, int)):
            value = np.ones_like(A[:,n])*value
        A[:,n] = value

    beta = np.linalg.inv(A.T @ A) @ A.T @ Y
    return beta

x = np.linspace(0, 1, 100)
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

beta, exponents = least_squares_poly(x, y, 2)

Z2 = np.zeros_like(Z)
for b,e in zip(beta, exponents):
    Z2 = Z2 + b*(X**e[0])*(Y**e[1])

fig = plt.figure()
ax = fig.gca(projection="3d")
surf = ax.plot_surface(X, Y, Z2, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

fxn_list = ["1", "x", "y", "x*y", "x**2", "y**2"]
variables = ["x", "y"]

beta = least_squares_functions(x, y, fxn_list, variables)

x = X
y = Y

Z2 = np.zeros_like(X)
for b,fxn in zip(beta, fxn_list):
    Z2 = Z2 + b*eval(fxn)

fig = plt.figure()
ax = fig.gca(projection="3d")
surf = ax.plot_surface(X, Y, Z2, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
