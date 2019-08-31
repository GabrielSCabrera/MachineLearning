from itertools import permutations
import matplotlib.pyplot as plt
import numpy as np

class Regression():

    def __init__(self, X, Y, dtype = np.float64):
        """
            ---PURPOSE------------------------------------

            Initialized a <Regression> object

            ---INPUT--------------------------------------

            X           2-D Array of numbers with shape (N, p)
            y           Array of numbers with shape (N,) or (N, 1)

            ---OPTIONAL-INPUT-----------------------------

            dtype       Datatype - see NumPy's documentation for <dtype>

        """

        error_msg = (f" in initialization of <Regression> object must be a "
                     f"NumPy array containing only numbers.")

        # Attempting to convert X to a NumPy array

        try:
            X = np.array(X, dtype = dtype)
        except ValueError:
            raise ValueError("\n\nParameter <X>" + error_msg)

        # Attempting to convert Y to a NumPy array

        try:
            Y = np.array(Y, dtype = dtype)
        except ValueError:
            raise ValueError("\n\nParameter <Y>" + error_msg)

        # Checking that X is 2-D

        if len(X.shape) != 2:
            error_msg = (f"\n\nParameter <X> in initialization of "
                         f"<Regression> object must be two-dimensional\n\t"
                         f"X.shape = {X.shape}")
            raise ValueError(error_msg)

        # Checking that X and Y are dimensionally compatible

        if X.shape[0] != Y.shape[0]:
            error_msg = (f"\n\nParameters <X> and <Y> in initialization of "
                         f"<Regression> object must have compatible "
                         f"matrix dimensions.  \n\n\tX.shape = {X.shape} -->"
                         f" suggest changing to ({Y.shape[0]}, {X.shape[1]})"
                         f"\n\tY.shape = {Y.shape}")
            raise ValueError(error_msg)

        # If Y is a 2-D column-vector, converts it to a 1-D row-vector

        if len(Y.shape) == 2 and Y.shape[1] == 1 and len(Y.shape) == 2:
            Y = Y[:,0]

        # Checking that Y is 1-D

        if len(Y.shape) != 1:
            error_msg = (f"\n\nParameter <Y> in initialization of "
                         f"<Regression> object must be one dimensional."
                         f"\n\n\tY.shape = {Y.shape}")
            raise ValueError(error_msg)

        self._X = X
        self._Y = Y
        self._dtype = dtype
        self._complete = False
        self._N = X.shape[0]
        self._p = X.shape[1]

    def _check_regr(self):
        """
            Helper method to inform user that the object instance has not yet
            had regression applied to it in a situation where completion is
            necessary.
        """
        if self._complete is False:
            error_msg = (f"\n\n<Regression> object cannot be displayed; must "
                         f"first select a regression method.\n\nExample:\n\n"
                         f"IN[1]\tfoo = Regression([[1],[2],[3]], [3,5,8])\n\t"
                         f"foo.poly(degree = 2) # Selection of the regression"
                         f" method.\n\tprint(foo)\n\nOUT[1]\t")
            raise Exception(error_msg)

    def poly(self, degree, alpha = None):
        """
            ---PURPOSE------------------------------------

            Implements regression for a multidimensional polynomial of the
            given <degree>.

            Can also implement ridge regression by passing a number greater
            than zero to parameter <alpha>.

            ---INPUT--------------------------------------

            degree      Integer greater than zero

            ---OPTIONAL-INPUT-----------------------------

            alpha       Real number greater than zero or None

            ---NOTES--------------------------------------

            When directly working with the vector of coefficients, be sure you
            know which coefficients correspond to each term in the polynomial.

            After running this method, the corresponding order of terms can be
            accessed via the <Regression.terms()> method.

        """

        # Checking that <degree> is an integer greater than zero
        try:
            if degree == int(degree) and degree > 0:
                degree = int(degree)
        except ValueError:
            error_msg = (f"\n\nParameter <degree> in <Regression.poly> "
                         f"must be an integer greater than zero\n\t"
                         f"type(degree) = {type(degree)}")
            raise ValueError(error_msg)

        # Checking that <alpha> is either a positive number of Nonetype
        if alpha is not None:
            try:
                if alpha > 0:
                    alpha = float(alpha)
                elif alpha <= 0:
                    raise ValueError()
            except TypeError:
                error_msg = (f"\n\nParameter <alpha> in <Regression.poly> "
                             f"must be a number greater than zero\n\t"
                             f"type(alpha) = {type(alpha)}")
                raise TypeError(error_msg)
            except ValueError:
                error_msg = (f"\n\nParameter <alpha> in <Regression.poly> "
                             f"must be a number greater than zero\n\t"
                             f"alpha = {alpha} --> suggest changing to: alpha "
                             f"= 1E-5")
                raise ValueError(error_msg)

        M = int(degree) + 1

        # Setting up all the cross terms of the polynomial
        powers = np.arange(0, M, 1)
        exponents = list(permutations(powers, self._p))

        # Including the non cross terms
        if self._p != 1:
            for power in powers:
                exponents.append(power*np.ones(self._p))

        # Excluding cross terms whose total is greater than <degree>
        if self._p != 1:
            expo_sum = np.sum(exponents, axis = 1)
            valid_idx = np.where(np.less_equal(expo_sum, degree))[0]
            exponents = np.array(exponents)
            exponents = exponents[valid_idx]
        else:
            exponents = np.array(exponents)

        # Creating the design matrix
        if self._p > 1:
            A = np.zeros((self._N, exponents.shape[0]))
            for n,exponent in enumerate(exponents):
                A[:,n] = np.prod(self._X**exponent, axis = 1)
        else:
            A = np.zeros((self._N, exponents.shape[0]))
            for n,exponent in enumerate(exponents):
                A[:,n] = self._X[:,0]**exponent

        # Implementing the least-squares method
        if alpha is None:
            # If ridge regression is not implemented
            self._beta = np.linalg.inv(A.T @ A) @ A.T @ self._Y
        else:
            # If ridge regression is implemented
            self._beta = A.T @ A
            self._beta = np.linalg.inv(self._beta + alpha*\
            np.identity((self._beta).shape[0])) @ A.T @ self._Y

        self._terms = exponents

    def terms(self):
        """
            Returns the terms for each respective element in the coefficient
            of vectors as a list of evaluatable strings
        """
        self._check_regr()

    def __str__(self):
        self._check_regr()
        return str_out


if __name__ == "__main__":
    r = Regression([[1,2],[2.2,1.1],[3,1.7],[3.2,1.1]], [3,5,8,7])
    r.poly(2, alpha = 0.01)
