from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

np.random.seed(69420666)

def least_squares(x, y, fxn_list):
    """
        ---PURPOSE------------------------------------

        Finds the vector of coefficients for a given set of functions and
        set of x and y points. The intercept/bias is automatically included.

        ---INPUT--------------------------------------

        x           Array of shape (N,)
        y           Array of shape (N,)
        fxnlist     List of M strings

        ---OUTPUT-------------------------------------

        beta        NumPy array of shape (M+1,)

        ---NOTES--------------------------------------

        The strings in parameter <fxnlist> should be composed of mathematically
        evaluatable expressions.

        For example, to perform linear regression,
        we would pass the list ["x"]; for a quadratic equation, we would pass
        the list ["x", "x**2"].

        To match a trigonometric function of sines and cosines, we could
        pass ["np.sin(x)", "np.cos(x)"].

        (Tip: Make sure NumPy is imported in the above case!)
    """

    x = np.array(x)
    y = np.array(y)

    if x.shape != y.shape:
        msg = "\n\nArguments <x> and <y> in function <least_squares> must be of "
        msg += f"the same shape.  \nx.shape = {x.shape}\ny.shape = {y.shape}"
        raise Exception(msg)

    if len(x.shape) != 1:
        msg = "\n\nArguments <x> and <y> in function <least_squares> must be "
        msg += f"one dimensional.  \nx.shape = {x.shape}\ny.shape = {y.shape}"
        raise Exception(msg)

    N = x.shape[0]

    if not isinstance(fxn_list, (list, tuple)):
        msg = "\n\nArgument <fxn_list> in function <least_squares> must be a "
        msg += f"list or tuple.  \ntype(fxn_list) = {type(fxn_list)}"
        raise Exception(msg)

    for n,fxn in enumerate(fxn_list):
        if not isinstance(fxn, str):
            msg = "\n\nElements of <fxn_list> in function <least_squares> must"
            msg += f" be strings.  \ntype(fxn_list[{n}]) = {type(fxn_list[n])}"
            raise Exception(msg)
        else:
            try:
                eval(fxn)
            except NameError:
                msg = "\n\nElements of <fxn_list> in function <least_squares> "
                msg += " must be mathematically evaluatable."
                msg += f"\nfxn_list[{n}] = \"{fxn}\" is an invalid expression."
                msg += f"\n\nTroubleshooting:\n\tIs {fxn} in the standard "
                msg += "library?  If not, be sure to import it!"
                raise Exception(msg)

    if "1" not in fxn_list:
        fxn_list = ["1"] + list(fxn_list)

    M = len(fxn_list)

    X = np.tile(x, reps = (M,1)).T
    A = np.zeros((N, M))

    for n,fxn in enumerate(fxn_list):
        x = X[:,n]
        A[:,n] = eval(fxn)

    Y = np.array([np.matmul(A.T, y.T)]).T
    beta = sp.Matrix(np.hstack([np.matmul(A.T,A), Y])).rref()[0][:,-1]
    beta = np.array(beta)

    return beta.T[0]

x = np.random.rand(100)
y = 5*x*x+0.1*np.random.randn(100)

beta_manual = least_squares(x,y, ["x", "x**2"])

X = PolynomialFeatures(degree = 2).fit_transform(np.array([x]).T)
beta_sklearn = linear_model.LinearRegression().fit(X,y).coef_

print(beta_manual)
print(beta_sklearn)

x2 = np.linspace(0, 1, 1E3)
y2 = beta_manual[0] + beta_manual[1]*x2 + beta_manual[2]*x2**2
y3 = beta_sklearn[0] + beta_sklearn[1]*x2 + beta_sklearn[2]*x2**2

plt.plot(x2, y2)
plt.plot(x2, y3)
plt.plot(x,y, "r.", ms = 3)
plt.legend(["Manual Quadratic Regression", "Scikit-Learn Quadratic Regression", "Datapoints"])
plt.xlim([np.min(x), np.max(x)])
plt.xlabel("Dimensionless $x$")
plt.ylabel("Dimensionless $y$")
plt.show()
