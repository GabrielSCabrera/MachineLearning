from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(69420666)

def ridge_method(x, y, L, degree):
    """
        ---PURPOSE------------------------------------

        Finds the vector of coefficients for a polynomial of varying degree and
        a set of x and y points via ridge regression.

        ---INPUT--------------------------------------

        x           Array of shape (N,)
        y           Array of shape (N,)
        L           Number/Array of Numbers >= 0
        degree      Integer greater than zero

        ---OUTPUT-------------------------------------

        beta        NumPy array of shape (degree,)

        ---NOTES--------------------------------------

        L is the lambda parameter used in ridge regression

    """

    X = np.array([x]).T
    y = np.array([y]).T

    N = X.shape[0]

    if X.shape != y.shape:
        msg = "\n\nArguments <x> and <y> in function <ridge_method> must be of "
        msg += f"the same shape.  \n\n\tx.shape = {x.shape}\ny.shape = {y.shape}"
        raise Exception(msg)

    if len(x.shape) != 1:
        msg = "\n\nArguments <x> and <y> in function <ridge_method> must be "
        msg += f"one dimensional.  \n\n\tx.shape = {x.shape}\ny.shape = {y.shape}"
        raise Exception(msg)

    try:
        if np.less(L, 0).any():
            msg = "\n\nArgument <L> in function <ridge_method> must be a number "
            msg += f" or array of numbers greater than or equal to zero."
            raise Exception(msg)
    except ValueError:
        msg = "\n\nArgument <L> in function <ridge_method> must be a "
        msg += f"number or array of numbers."
        raise Exception(msg)

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

    X = np.hstack([np.ones_like(X), X])
    M = degree+1

    X = np.tile(x, reps = (M,1)).T
    A = np.zeros((N, M))

    for n in range(M):
        x = X[:,n]
        A[:,n] = x**n

    beta_ridge = np.matmul(A.T, A)
    beta_ridge = np.linalg.inv(beta_ridge + L*np.identity(beta_ridge.shape[0]))
    beta_ridge = np.matmul(beta_ridge, A.T)
    beta_ridge = np.matmul(beta_ridge, y)

    return beta_ridge.T[0]

def ridge_variance(x, y, L, degree):
    """
        ---PURPOSE------------------------------------

        Returns the variance of parameters for a vector of coefficients.

        ---INPUT--------------------------------------

        x           Array of shape (N,)
        y           Array of shape (N,)
        L           Number/Array of Numbers >= 0
        degree      Integer greater than zero

        ---OUTPUT-------------------------------------

        variance        NumPy array of shape: np.diags(x).shape

        ---NOTES--------------------------------------

        L is the lambda parameter used in ridge regression

    """

    X = np.array([x]).T
    y = np.array([y]).T

    N = X.shape[0]

    if X.shape != y.shape:
        msg = ( f"\n\nArguments <x> and <y> in function <ridge_method> must be"
                f" of the same shape.  \n\n\tx.shape = {x.shape}\ny.shape = "
                f"{y.shape}" )
        raise Exception(msg)

    if len(x.shape) != 1:
        msg = "\n\nArguments <x> and <y> in function <ridge_method> must be "
        msg += f"one dimensional.  \n\n\tx.shape = {x.shape}\ny.shape = {y.shape}"
        raise Exception(msg)

    try:
        if np.less(L, 0).any():
            msg = "\n\nArgument <L> in function <ridge_method> must be a number "
            msg += f" or array of numbers greater than or equal to zero."
            raise Exception(msg)
    except ValueError:
        msg = "\n\nArgument <L> in function <ridge_method> must be a "
        msg += f"number or array of numbers."
        raise Exception(msg)

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

    X = np.hstack([np.ones_like(X), X])
    M = degree+1

    X = np.tile(x, reps = (M,1)).T
    A = np.zeros((N, M))

    for n in range(M):
        x = X[:,n]
        A[:,n] = x**n

    variance_ridge = np.matmul(A.T, A)
    variance_ridge = np.linalg.inv(variance_ridge + L*np.identity(variance_ridge.shape[0]))
    variance_ridge = np.diagonal(variance_ridge)

    return variance_ridge

def plot_regressions(noise, L_values, plot = True):

    if plot is True:
        fig, ax = plt.subplots(2)

    x = np.random.rand(100, 1)
    y = 5*x*x+noise*np.random.randn(100, 1)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.33,
    random_state = 42)
    A_train = PolynomialFeatures(degree = 2).fit_transform(X_train)

    if plot is True:
        x2 = np.linspace(np.min(X_train), np.max(X_train), 5E2)

    if plot is True:
        legend0 = ["Datapoints"]
        legend1 = ["Datapoints"]

    if plot is True:
        ax[0].plot(x, y, "r.", ms = 3)

        ax[1].plot(x, y, "r.", ms = 3)

    variances = []

    if plot is True:
        print(f"\nNoise Coefficient: {noise:g}")
    for L in L_values:
        beta_ridge = ridge_method(X_train.T[0], y_train.T[0], L, 2)
        if plot is True:
            y2 = beta_ridge[0] + beta_ridge[1]*x2 + beta_ridge[2]*x2**2
            ax[0].plot(x2, y2)

        beta_ridge_sklearn = Ridge(alpha = L).fit(A_train, y_train).coef_[0]
        if plot is True:
            y3 = beta_ridge_sklearn[0] + beta_ridge_sklearn[1]*x2 + beta_ridge_sklearn[2]*x2**2
            ax[1].plot(x2, y3)

        variance = ridge_variance(X_train.T[0], y_train.T[0], L, 2)
        if plot is True:
            print(f"lambda = {L:<6.2g} > variance = {variance}")
        variances.append((variance, L))

        if plot is True:
            legend0.append(f"Manual $\\lambda = {L:g}$")
            legend1.append(f"Sklearn $\\lambda = {L:g}$")

    if plot is True:
        ax[0].legend(legend0, loc = "upper left", frameon = False)
        ax[1].legend(legend1, loc = "upper left", frameon = False)

        ax[0].set_xlim([np.min(x), np.max(x)])
        ax[0].set_xlabel("Dimensionless $x$")
        ax[0].set_ylabel("Dimensionless $y$")

        ax[1].set_xlim([np.min(x), np.max(x)])
        ax[1].set_xlabel("Dimensionless $x$")
        ax[1].set_ylabel("Dimensionless $y$")

        plt.show()
    return variances

def plot_variances(variances):
    legend = []
    for vars_L, noise in variances:
        vars = []
        Ls = []
        for var, L in vars_L:
            vars.append(var)
            Ls.append(L)
        vars = np.array(vars)
        Ls = np.array(Ls)
        for i in range(vars.shape[1]):
            plt.plot(Ls, vars[:,i])
            leg_element = r"$X_{" + str(i+1) + r"," + str(i+1) + r"}$"
            legend.append(leg_element)
        plt.xlabel("$\\lambda$")
        plt.ylabel(f"Variance (noise = {noise:6.2g})")
        plt.xlim(np.min(Ls), np.max(Ls))
        plt.legend(legend)
        plt.show()

noise = np.logspace(-3, 2, 5)
variances = []

L_values = np.logspace(-5, 1, 3)
for n in noise:
    plot_regressions(n, L_values)

L_values = np.linspace(1E-10, 10, 100)
for n in noise:
    variances.append((plot_regressions(n, L_values, False), n))

plot_variances(variances)
