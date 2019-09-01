from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import matplotlib

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
                         f" method")
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
            exponents = np.array(exponents, dtype = np.int64)
            exponents = exponents[valid_idx]
        else:
            exponents = np.array(exponents, dtype = np.int64)

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
            try:
                self._beta = np.linalg.inv(A.T @ A) @ A.T @ self._Y
            except np.linalg.LinAlgError:
                error_msg = (f"\n\nThe design matrix product could not be "
                             f"inverted because it is a singular matrix.\n\n\t"
                             f"(XᵀX)⁻¹ = undefined\n\nTry setting parameter "
                             f"<alpha> in <Regression.poly> to 1E-4; this "
                             f"implements ridge regression, and may lead to an"
                             f" approximate solution.")

                raise np.linalg.LinAlgError(error_msg)
        else:
            # If ridge regression is implemented
            self._beta = A.T @ A
            try:
                self._beta = np.linalg.inv(self._beta + alpha*\
                np.identity((self._beta).shape[0])) @ A.T @ self._Y
            except np.linalg.LinAlgError:
                error_msg = (f"\n\nThe design matrix product could not be "
                             f"inverted because it is a singular matrix.\n\n\t"
                             f"(XᵀX + {alpha:g}*I)⁻¹ = undefined\n\nTry "
                             f"increasing parameter <alpha> to find a "
                             f"non-singular matrix.")
                raise np.linalg.LinAlgError(error_msg)

        self._readable, self._terms = self._poly_str(exponents, self._beta)
        self._complete = True

    def _poly_str(self, exponents, beta):
        """
            ---PURPOSE------------------------------------

            Takes a set of exponents and a vector of coefficients and
            uses these to create a list of strings which are human-readable.

            Should not be accessed by users.

            ---INPUT--------------------------------------

            exponents       2-D NumPy array of shape (N, d)
            beta            1-D Numpy array of shape (N,)

            ---OUTPUT-------------------------------------

            readable        String containing human-readable polynomial
            terms           List of evaluatable strings of length N

            ---NOTES--------------------------------------

            Uses variables:
                                x, y, z, w, v, u

            for up to six dimensional functions, in the given order.
            For d > 6 dimensional functions, generalizes to:

                            x_i; i = 1, 2, ... , d

            Example:

            IN[1]       foo = Regression._poly_str()
                        foo.poly(1)
                        print(foo)

            OUT[1]      F(x, y) = -1.38*y - 1.22*x + 9.03

        """

        all_var_chars = ["x", "y", "z", "w", "v", "u"]

        N_terms = exponents.shape[0]
        N_vars = exponents.shape[1]

        if N_vars > 6:
            var_chars = [f"x_{i+1}" for i in range(N_vars)]
        else:
            var_chars = all_var_chars[:N_vars]

        self._var_chars = var_chars

        terms = []
        if N_vars == 1:
            equation = "f("
        else:
            equation = "F("

        if N_vars <= 6:
            for n,c in enumerate(var_chars):
                if n > 0:
                    equation = f"{equation}, "
                equation = f"{equation}{c}"
        else:
            equation = f"{equation}x_i; i = 1,2,...,{N_vars}"

        equation = f"{equation}) = "

        for m,(e,b) in enumerate(zip(exponents, beta)):
            b_sign = np.sign(b)
            factors = [f"{abs(b):.3g}"]
            for n,i in enumerate(e):
                if i != 0:
                    factors.append(f"{var_chars[n]}")
                if i > 1:
                    factors[-1] += f"**{i:d}"

            if len(factors) == 0:
                factors.append("")
            elif len(factors) > 1:
                for n,f in enumerate(factors):
                    if "**" in f:
                        factors[n] = "(" + f + ")"

            factors = "*".join(factors)

            if b_sign == -1:
                terms.append(f"-{factors}")
            else:
                terms.append(factors)

            if m > 0:
                if b_sign == 1:
                    factors = f" + {factors}"
                elif b_sign == -1:
                    factors = f" - {factors}"
            elif b_sign == -1:
                factors = f"-{factors}"

            equation = f"{equation}{factors}"

        return equation, terms

    def terms(self):
        """
            Returns the terms for each respective element in the coefficient
            of vectors as a list of evaluatable strings
        """
        self._check_regr()
        return self._terms

    def __str__(self):
        self._check_regr()
        return self._readable

    def plot(self, detail = 0.5, xlabel = None, ylabel = None, zlabel = None,
    savename = None):
        """
            ---PURPOSE------------------------------------

            To plot the input data and regression function side-by-side using
            matplotlib.pyplot.

            ---OPTIONAL-INPUT-----------------------------

            detail          Number in range [0,1]
            xlabel          String or None
            ylabel          String or None
            zlabel          String or None
            savename        String or None

            ---NOTES--------------------------------------

            Only supports 2-D and 3-D regressions, will raise a ValueError
            if an attempt is made to plot a 4+ dimensional system.

            Parameter <detail> determines how detailed the regression plot will
            be.  The lowest value 0 will yield a quickly renderable plot, while
            the highest value 1 will yield a very smooth regression curve or
            surface.  Higher values may not run well on lower-end systems.

            If creating a 2-D plot, passing a value to <zlabel> will raise a
            ValueError.

        """
        self._check_regr()

        if self._p not in [1,2]:
            error_msg = (f"\n\nAttempting to call method <Regression.plot()> "
                         f"on a {self._p+1}-D set of datapoints.\n\nOnly 2-D "
                         f"and 3-D datasets are supported.")
            raise ValueError(error_msg)

        labels = [xlabel, ylabel, zlabel]
        label_names = ["xlabel", "ylabel", "zlabel"]

        if savename is not None and not isinstance(savename, str):
            error_msg = (f"\n\nParameter <savename> in method "
                         f"<Regression.plot> must be a string.\n\t"
                         f"type(savename) = {type(savename)}")
            raise TypeError(error_msg)

        error_msg = " in method <Regression.plot> must be a string.\n\t"
        for n,(label, name) in enumerate(zip(labels, label_names)):

            if n == 2 and self._p != 2 and label is not None:
                error_msg = (f"\n\nCannot use parameter <zlabel> in method "
                             f"<Regression.plot> when the dataset is 2-D."
                             f" Attempted to pass:\n\tzlabel = \"{label}\"")
                raise TypeError(error_msg)

            elif not isinstance(label, str) and label is not None:
                error_msg += f"type({name}) = {type(label)}"
                raise TypeError(f"\n\nParameter <{name}>{error_msg}")

        try:
            detail = float(detail)
            if not 0 <= detail <= 1:
                raise ValueError()
        except TypeError:
            error_msg = (f"\n\nParameter <detail> in method <Regression.plot> "
                         f"must be a number in range [0,1]\n\tdetail = "
                         f"{detail} --> suggest changing to: detail = 0.5")
            raise TypeError(error_msg)
        except ValueError:
            error_msg = (f"\n\nParameter <detail> in method <Regression.plot> "
                         f"must be a number in range [0,1]\n\tdetail = "
                         f"{detail} --> suggest changing to: detail = 0.5")
            raise ValueError(error_msg)

        local_scope = locals()

        if savename is None:
            matplotlib.rcParams.update({'font.size': 22})

        if self._p == 1:
            N_max = 1E5
            N_points = int(N_max*np.log10(detail+1))+50
            x = np.linspace(np.min(self._X[:,0]), np.max(self._X[:,0]), N_points)
        elif self._p == 2:
            N_max = 35
            N_points = int((N_max*np.log10(detail+1))**2)+5
            x = np.linspace(np.min(self._X[:,0]), np.max(self._X[:,0]), N_points)
            y = np.linspace(np.min(self._X[:,1]), np.max(self._X[:,1]), N_points)
            x,y = np.meshgrid(x,y)

        F = np.zeros_like(x)
        for term in self._terms:
            F = F + eval(term)

        if self._p == 1:
            plt.plot(self._X[:,0], self._Y, "x", ms = 10)
            plt.plot(x, F, "r-")
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend(["Training Data", "Regression Curve"])
            plt.xlim(np.min(self._X[:,0]), np.max(self._X[:,0]))
            figManager = plt.get_current_fig_manager()
            figManager.full_screen_toggle()
            if savename is not None:
                plt.savefig(savename, dpi = 250)
                plt.close()
            else:
                plt.show()
        elif self._p == 2:
            fig = plt.figure()
            ax = fig.gca(projection="3d")
            fig.set_size_inches(8, 6)
            fig.tight_layout()

            surf = ax.scatter(self._X[:,0], self._X[:,1], self._Y, s = 100,
            marker = "x")

            surf = ax.plot_surface(x, y, F, cmap=cm.copper, alpha = 0.85,
            linewidth=0, antialiased=False, rcount = 116, ccount = 116)

            if xlabel is not None:
                ax.set_xlabel("\n\n\n" + xlabel + "\n", linespacing = 3)
            if ylabel is not None:
                ax.set_ylabel("\n\n\n" + ylabel + "\n", linespacing = 3)
            if zlabel is not None:
                ax.set_zlabel("\n\n\n" + zlabel + "\n", linespacing = 3)

            # fig.colorbar(surf, shrink=0.5, aspect=5)
            figManager = plt.get_current_fig_manager()
            figManager.full_screen_toggle()

            if savename is not None:
                plt.savefig(savename, dpi = 250)
                plt.close()
            else:
                plt.show()
