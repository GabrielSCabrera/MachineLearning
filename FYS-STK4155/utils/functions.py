import numpy as np

def least_squares_functions(x, y, fxn_list):
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
        msg = "\n\nArguments <x> and <y> in function <least_squares_functions> must be of "
        msg += f"the same shape.  \n\n\tx.shape = {x.shape}\ny.shape = {y.shape}"
        raise Exception(msg)

    if len(x.shape) != 1:
        msg = "\n\nArguments <x> and <y> in function <least_squares_functions> must be "
        msg += f"one dimensional.  \n\n\tx.shape = {x.shape}\ny.shape = {y.shape}"
        raise Exception(msg)

    N = x.shape[0]

    if not isinstance(fxn_list, (list, tuple)):
        msg = "\n\nArgument <fxn_list> in function <least_squares_functions> must be a "
        msg += f"list or tuple.  \n\n\ttype(fxn_list) = {type(fxn_list)}"
        raise Exception(msg)

    for n,fxn in enumerate(fxn_list):
        if not isinstance(fxn, str):
            msg = "\n\nElements of <fxn_list> in function <least_squares_functions> must"
            msg += f" be strings.  \n\n\ttype(fxn_list[{n}]) = {type(fxn_list[n])}"
            raise Exception(msg)
        else:
            try:
                eval(fxn)
            except NameError:
                msg = "\n\nElements of <fxn_list> in function <least_squares_functions> "
                msg += " must be mathematically evaluatable."
                msg += f"\n\n\tfxn_list[{n}] = \"{fxn}\" is an invalid expression."
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

    beta = np.matmul(np.linalg.inv(np.matmul(A.T, A)),A.T)
    beta = np.matmul(beta, y)

    return beta.T[0]

def least_squares_poly(x, y, degree):
    """
        ---PURPOSE------------------------------------

        Finds the vector of coefficients for a polynomial of varying degree and
        a set of x and y points.

        ---INPUT--------------------------------------

        x           Array of shape (N,)
        y           Array of shape (N,)
        degree      Integer greater than zero

        ---OUTPUT-------------------------------------

        beta        NumPy array of shape (degree,)

    """

    x = np.array(x)
    y = np.array(y)

    if x.shape != y.shape:
        msg = "\n\nArguments <x> and <y> in function <least_squares_poly> must be of "
        msg += f"the same shape.  \n\n\tx.shape = {x.shape}\ny.shape = {y.shape}"
        raise Exception(msg)

    if len(x.shape) != 1:
        msg = "\n\nArguments <x> and <y> in function <least_squares_poly> must be "
        msg += f"one dimensional.  \n\n\tx.shape = {x.shape}\ny.shape = {y.shape}"
        raise Exception(msg)

    N = x.shape[0]

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

    M = degree+1

    X = np.tile(x, reps = (M,1)).T
    A = np.zeros((N, M))

    for n in range(M):
        x = X[:,n]
        A[:,n] = x**n

    beta = np.matmul(np.linalg.inv(np.matmul(A.T, A)),A.T)
    beta = np.matmul(beta, y)

    return beta.T[0]

def ridge_method_poly(x, y, L, degree):
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
        msg = "\n\nArguments <x> and <y> in function <ridge_method_poly> must be of "
        msg += f"the same shape.  \n\n\tx.shape = {x.shape}\ny.shape = {y.shape}"
        raise Exception(msg)

    if len(x.shape) != 1:
        msg = "\n\nArguments <x> and <y> in function <ridge_method_poly> must be "
        msg += f"one dimensional.  \n\n\tx.shape = {x.shape}\ny.shape = {y.shape}"
        raise Exception(msg)

    try:
        if np.less(L, 0).any():
            msg = "\n\nArgument <L> in function <ridge_method_poly> must be a number "
            msg += f" or array of numbers greater than or equal to zero."
            raise Exception(msg)
    except ValueError:
        msg = "\n\nArgument <L> in function <ridge_method_poly> must be a "
        msg += f"number or array of numbers."
        raise Exception(msg)

    try:
        if degree == int(degree) and degree > 0:
            degree = int(degree)
        else:
            msg = "\n\nArgument <degree> in function <ridge_method_poly> must be an "
            msg += f"integer greater than zero.  \ndegree = {degree}"
            raise Exception(msg)
    except ValueError:
        msg = "\n\nArgument <degree> in function <ridge_method_poly> must be a "
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

def ridge_variance_poly(x, y, L, degree):
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
        msg = ( f"\n\nArguments <x> and <y> in function <ridge_variance_poly> must be"
                f" of the same shape.  \n\n\tx.shape = {x.shape}\ny.shape = "
                f"{y.shape}" )
        raise Exception(msg)

    if len(x.shape) != 1:
        msg = "\n\nArguments <x> and <y> in function <ridge_variance_poly> must be "
        msg += f"one dimensional.  \n\n\tx.shape = {x.shape}\ny.shape = {y.shape}"
        raise Exception(msg)

    try:
        if np.less(L, 0).any():
            msg = "\n\nArgument <L> in function <ridge_variance_poly> must be a number "
            msg += f" or array of numbers greater than or equal to zero."
            raise Exception(msg)
    except ValueError:
        msg = "\n\nArgument <L> in function <ridge_variance_poly> must be a "
        msg += f"number or array of numbers."
        raise Exception(msg)

    try:
        if degree == int(degree) and degree > 0:
            degree = int(degree)
        else:
            msg = "\n\nArgument <degree> in function <ridge_variance_poly> must be an "
            msg += f"integer greater than zero.  \ndegree = {degree}"
            raise Exception(msg)
    except ValueError:
        msg = "\n\nArgument <degree> in function <ridge_variance_poly> must be a "
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
