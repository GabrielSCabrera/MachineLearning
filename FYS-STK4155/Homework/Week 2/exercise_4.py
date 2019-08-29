from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

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

x = np.random.rand(100, 1)
y = 5*x*x+0.1*np.random.randn(100, 1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

beta_ridge = ridge_method(X_train.T[0], y_train.T[0], 10, 2)
print(beta_ridge)
