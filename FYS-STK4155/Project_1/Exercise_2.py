import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

def least_squares_functions(points, functions, return_coefficients = False,
include_one = True):
    '''Uses the method of least squares to match a set of points

    <points> should be an array of dimensions (n, 2), where n is the number of
    points we are trying to match.  <functions> should be a string or array of
    strings that can be numpy evaluated mathematically in terms of a variable
    <x> - do not include coefficients. <return_coefficients> determines whether
    or not the list of coefficients is returned - if false, simply returns the
    function'''

    points= np.array(points)

    if len(points.shape) == 2 and points.shape[0] == 2 and points.shape[1] != 2:
        points= points.transpose()
    point_error_msg= 'Invalid format for argument <points>'
    assert len(points.shape) == 2 and points.shape[1] == 2, point_error_msg

    if isinstance(functions, str):
        if include_one == True:
            functions= ['1', functions]
        else:
            functions= [functions]
    functions_error_msg= 'Invalid format for argument <functions>'
    assert isinstance(functions, (list, tuple)), functions_error_msg
    if isinstance(functions, tuple):
        functions= list(functions)
    if include_one == True and '1' not in functions:
        functions= ['1'] + functions

    math_error_msg= 'Non-evaluable function in list <functions>'

    X, y= np.tile(points[:,0], reps= (len(functions),1)).transpose(), points[:,1]
    A= np.zeros((len(X), len(functions)))
    for n,f in enumerate(functions):
        x= X[:,n]
        try:
            A[:,n]= eval(f)
        except ValueError:
            assert False, math_error_msg

    A= np.asmatrix(A)
    AT= A.transpose()
    Y= AT*np.asmatrix(y).transpose()
    coefficients= sp.Matrix(np.concatenate((AT*A, Y), axis= -1)).rref()[0][:,-1]
    coefficients= np.array(coefficients)

    if return_coefficients == False:
        eval_string= ''
        for n,(c,f) in enumerate(zip(coefficients, functions)):
            eval_string+= '(%f*(%s))'%(c,f)
            if n < len(functions) - 1:
                eval_string+= '+'
        def f(x):
            return eval(eval_string)
        return f
    return coefficients

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

    Y = np.matmul(A.T, y.T)
    beta = sp.Matrix(np.vstack([np.matmul(A.T,A), Y])).rref()[0][:,-1]
    beta = np.array(beta)

    print(beta)

X = np.random.rand(100,1)
Y = 5*X*X+0.1*np.random.randn(100,1)

x = X[:,0]
y = Y[:,0]

beta = least_squares(x,y, ["1", "np.cos(x)", "x**2"])
beta = least_squares_functions([x,y], ["x", "x**2"], True)

# print(beta)

x2 = np.linspace(0, 1, 1E3)
y2 = beta[0] + beta[1]*x2 + beta[2]*x2**2

# plt.plot(x,y, "b.")
# plt.plot(x2, y2)
# plt.show()
