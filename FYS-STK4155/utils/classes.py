from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import matplotlib
import warnings
from sklearn import linear_model as skl

np.random.seed(69420666)

class Regression():

    def __init__(self, X, Y, dtype = np.float64):
        """
            ---PURPOSE------------------------------------

            Initializes a <Regression> object

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

        self._X_backup = X
        self._Y_backup = Y

        self._X = X
        self._Y = Y
        self._dtype = dtype
        self._complete = False
        self._predicted = False
        self._split = False
        self._N = self._X.shape[0]
        self._p = self._X.shape[1]

        # Cleans out all attributes in case of reset
        if not hasattr(self, '_dir_backup'):
            self._dir_backup = self.__dir__().copy
        else:
            for var in self.__dir__():
                if var not in self._dir_backup() and var != "_dir_backup":
                    delattr(self, var)

    def split(self, test_size):
        """
            ---PURPOSE------------------------------------

            Takes the dataset given by X, Y in __init__ and splits it into
            training data and testing data automatically.

            ---OPTIONAL-INPUT-----------------------------

            test_size           The percentage (0,100) of that data that
                                should be allocated to the test input

            ---OUTPUT-------------------------------------

            arg_idx             2-D list

            ---NOTES--------------------------------------

            The returned list <arg_idx> contains two sublists; the first
            sublist contains the indices of the training set, and the second
            contains the indices of the test set, as were randomly selected.
        """

        self._check_not_regr("split")

        if self._split is True:
            error_msg = (f"\n\nInvalid usage of <Regression> object; cannot "
                         f"split the dataset into training and testing data "
                         f"multiple times with <Regression.split>.")
            raise Exception(error_msg)

        # Attempting to convert X to a NumPy array
        error_msg = (f"Parameter <test_size> in method <Regression.split> "
                     f" must be a number between (and not including) 0 and 100"
                     f".\n\t")
        try:
            test_size = float(test_size)
            if test_size <= 0 or test_size >= 100:
                raise ValueError()
        except TypeError:
            error_msg += f"type(test_size) = {type(test_size)}"
            raise TypeError(error_msg)
        except ValueError:
            error_msg += (f"test_size = {test_size} --> suggest changing to "
                          f"test_size = 33")
            raise ValueError(error_msg)

        N_test = int(round(self._N*test_size*0.01))
        self._test_idx = np.random.choice(a = N_test, size = N_test, replace = False)
        self._X_test = self._X[self._test_idx]
        self._Y_test = self._Y[self._test_idx]
        self._X = np.delete(self._X, self._test_idx, axis = 0)
        self._Y = np.delete(self._Y, self._test_idx)
        self._split = True


        arg_idx = []
        arg_idx.append(list(np.setdiff1d(np.arange(0, self._N), self._test_idx)))
        arg_idx.append(list(self._test_idx))

        self._N = self._X.shape[0]
        self._p = self._X.shape[1]

        return arg_idx

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

        self._check_not_regr("poly")

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

        A, exponents = self._design(self._X, degree, "poly")

        # Implementing the least-squares method
        if alpha is None:
            # If ridge regression is not implemented
            try:
                self._beta = np.linalg.inv(A.T @ A)
                self._variance = np.diag(self._beta)
                self._beta = self._beta @ A.T @ self._Y
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
                np.identity((self._beta).shape[0]))
                self._variance = np.diagonal(self._beta)
                self._beta = self._beta @ A.T @ self._Y
            except np.linalg.LinAlgError:
                error_msg = (f"\n\nThe design matrix product could not be "
                             f"inverted because it is a singular matrix.\n\n\t"
                             f"(XᵀX + {alpha:g}*I)⁻¹ = undefined\n\nTry "
                             f"increasing parameter <alpha> to find a "
                             f"non-singular matrix.")
                raise np.linalg.LinAlgError(error_msg)

        self._readable, self._terms = self._poly_str(exponents, self._beta)
        self._complete = True
        self._exponents = exponents
        self._degree = degree

    def lasso(self, degree, alpha):
        """
            ---PURPOSE------------------------------------

            Implements lasso regression for a multidimensional polynomial.

            ---INPUT--------------------------------------

            degree      Integer over 0
            alpha       Real number greater than zero, or None

            ---NOTES--------------------------------------

            When directly working with the vector of coefficients, be sure you
            know which coefficients correspond to each term in the polynomial.

            After running this method, the corresponding order of terms can be
            accessed via the <Regression.terms()> method.
        """

        self._check_not_regr("lasso")

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

        A, exponents = self._design(self._X, degree, "lasso")

        clf = skl.Lasso(alpha=alpha, max_iter = 2E3)
        clf.fit(A, self._Y)
        beta = clf.coef_
        self._beta = beta
        self._beta[0] = clf.intercept_

        self._predicted = True
        self._readable, self._terms = self._poly_str(exponents, self._beta)
        self._complete = True
        self._exponents = exponents
        self._degree = degree

    def lasso_manual(self, degree, alpha, itermax = 500, tol = 2E-3):
        """
            ---PURPOSE------------------------------------

            Implements lasso regression for a multidimensional polynomial.

            ---INPUT--------------------------------------

            alpha       Real number greater than zero, or None

            ---OPTIONAL-INPUT-----------------------------

            itermax     Integer value
            tol         Real number

            ---NOTES--------------------------------------

            When directly working with the vector of coefficients, be sure you
            know which coefficients correspond to each term in the polynomial.

            After running this method, the corresponding order of terms can be
            accessed via the <Regression.terms()> method.

            Parameter <itermax> represents the maximum number of iterations
            that are allowed during the implementation of the lasso algorithm

            Parameter <tol> represents the smallest mean difference in the
            vector of coefficients <beta>  between steps -- this governs the
            point at which the lasso algorithm ceases to run.
        """

        self._check_not_regr("lasso")

        # Checking that <alpha> is a positive number
        if alpha is not None:
            try:
                if alpha > 0:
                    alpha = float(alpha)
                elif alpha <= 0:
                    raise ValueError()
            except TypeError:
                error_msg = (f"\n\nParameter <alpha> in <Regression.lasso> "
                             f"must be a number greater than zero\n\t"
                             f"type(alpha) = {type(alpha)}")
                raise TypeError(error_msg)
            except ValueError:
                error_msg = (f"\n\nParameter <alpha> in <Regression.lasso> "
                             f"must be a number greater than zero\n\t"
                             f"alpha = {alpha} --> suggest changing to: alpha "
                             f"= 1E-5")
                raise ValueError(error_msg)

        # Checking that <itermax> is either a positive number or Nonetype
        try:
            if itermax <= 0 or int(itermax) != itermax:
                raise ValueError()
            elif itermax > 0:
                itermax = int(itermax)
        except TypeError:
            error_msg = (f"\n\nParameter <itermax> in <Regression.lasso> "
                         f"must be an integer greater than zero\n\t"
                         f"type(itermax) = {type(itermax)}")
            raise TypeError(error_msg)
        except ValueError:
            error_msg = (f"\n\nParameter <itermax> in <Regression.lasso> "
                         f"must be an integer greater than zero\n\t"
                         f"itermax = {itermax} --> suggest changing to: "
                         f"itermax = 500")
            raise ValueError(error_msg)

        # Checking that <tol> is a positive number
        if tol is not None:
            try:
                if tol > 0:
                    tol = float(tol)
                elif tol <= 0:
                    raise ValueError()
            except TypeError:
                error_msg = (f"\n\nParameter <tol> in <Regression.lasso> "
                             f"must be a number greater than zero\n\t"
                             f"type(tol) = {type(tol)}")
                raise TypeError(error_msg)
            except ValueError:
                error_msg = (f"\n\nParameter <tol> in <Regression.lasso> "
                             f"must be a number greater than zero\n\t"
                             f"tol = {tol} --> suggest changing to: tol "
                             f"= 1E-3")
                raise ValueError(error_msg)

        A, exponents = self._design(self._X, degree, "lasso")

        z = np.sum(A**2, axis = 0)

        beta = np.zeros(A.shape[1])
        beta_new = np.zeros(A.shape[1])
        beta_old = beta.copy()

        dx_old = 0
        dx = None

        alpha = 0.01

        i = 0
        while (dx is None or dx > tol) and (i < itermax):
            i += 1
            for j in range(A.shape[1]):
                Y_hat = np.sum(beta*A, axis = 1) - (beta[j]*A[:,j])
                rho = np.sum(A[:,j]*(self._Y - Y_hat))
                if rho < -alpha/2:
                    beta[j] = (rho + alpha/2)/z[j]
                elif rho > alpha/2:
                    beta[j] = (rho - alpha/2)/z[j]
                else:
                    beta[j] = 0

            dx = np.mean(np.abs(beta - beta_old))
            dx_old = dx
            beta_old = beta.copy()

        self._beta = beta
        self._readable, self._terms = self._poly_str(exponents, self._beta)
        self._complete = True
        self._exponents = exponents

    def sigma(self):
        """
            ---PURPOSE------------------------------------

            Finds the variance sigma^2 of of original data vs.
            predicted data.

            ---OUTPUT-------------------------------------

            sigma2      Float

        """
        self._check_predict("sigma")
        sigma2 = 1/(self._N - self._p - 1) * np.sum((self._Y - self._Y_hat)**2)
        return sigma2

    def terms(self):
        """
            Returns the terms for each respective element in the coefficient
            of vectors as a list of evaluatable strings
        """
        self._check_regr("terms")
        return self._terms

    def predict(self, X = None):
        """
            ---PURPOSE------------------------------------

            Assuming that a regression has taken place, will predict the output
            given by a set of inputs in <X>, or if X = None, will use the input
            array self._X in its place.

            ---OPTIONAL-INPUT-----------------------------

            X           Numerical array of shape (M, p)

            ---OUTPUT-------------------------------------

            Y           Array of shape (M,)
        """

        self._check_regr("predict")

        if X is not None:
            A, e = self._design(X = X, degree = self._degree, method = "predict")
        else:
            A, e = self._design(X = self._X, degree = self._degree, method = "predict")

        Y_hat = A @ self._beta
        return Y_hat

    def variance(self, sigma = None, split = False):
        """
            ---PURPOSE------------------------------------

            Assuming that a regression has taken place, will return the
            variance in the vector of coefficients beta.

            ---OPTIONAL-INPUT-----------------------------

            sigma               Scalar value
            split               Boolean

            ---OUTPUT-------------------------------------

            var_beta            Array of shape (p,)

            ---NOTES--------------------------------------

            The parameter <sigma> is representative of the standard deviation
            in a Gaussian noise that is expected to exist in the dataset.  It
            is assumed that the dataset output is a function of the form:

                        y = f(x_1, x_2, ..., x_p) + N(0, sigma)

            Where N(mu, sigma) is a normal distribution with a mean value of mu
            and a standard deviation of sigma.
        """
        self._check_regr("variance")
        if split is True:
            self._check_split("variance")
        if sigma is None:
            sigma = self.sigma()
        else:
            sigma *= sigma
        return sigma*self._variance

    def mse(self, split = False):
        """
            ---PURPOSE------------------------------------

            Assuming that the dataset has been split, returns the mean-squared-
            error of the regression based on the testing data output and
            predicted values.

            ---OPTIONAL-INPUT-----------------------------

            split                   Boolean

            ---OUTPUT-------------------------------------

            mean_squared_error      Number of type float

            ---NOTES--------------------------------------

            The parameter <split> determines whether or not R² should be
            calculated for a split training-test set.  If True, it will use
            a test set, if False it will use the entire <X> array for both
            training and testing.
        """

        self._check_regr("mse")
        if split is True:
            self._check_split("mse")
            Y_hat = self.predict(self._X_test)
            mean_squared_error = np.mean((self._Y_test - Y_hat)**2)
        else:
            Y_hat = self.predict(self._X)
            mean_squared_error = np.mean((self._Y - Y_hat)**2)

        return mean_squared_error

    def bias(self, split = False):
        """
            ---PURPOSE------------------------------------

            Assuming that the dataset has been split, returns the bias of the
            regression based on the testing data output and predicted values.

            ---OPTIONAL-INPUT-----------------------------

            split                   Boolean

            ---OUTPUT-------------------------------------

            bias                    Number of type float

            ---NOTES--------------------------------------

            The parameter <split> determines whether or not R² should be
            calculated for a split training-test set.  If True, it will use
            a test set, if False it will use the entire <X> array for both
            training and testing.
        """

        self._check_regr("bias")
        if split is True:
            self._check_split("bias")
            Y_mean = np.mean(self.predict(self._X_test))
            bias = np.mean((self._Y_test - Y_mean)**2)
        else:
            Y_mean = np.mean(self.predict(self._X))
            bias = np.mean((self._Y - Y_mean)**2)

        return bias

    def r_squared(self, split = False):
        """
            ---PURPOSE------------------------------------

            Assuming that the dataset has been split, returns the R²
            score based on the testing data output and predicted values.

            ---OPTIONAL-INPUT-----------------------------

            split                   Boolean

            ---OUTPUT-------------------------------------

            r_squared               Number of type float

            ---NOTES--------------------------------------

            The parameter <split> determines whether or not R² should be
            calculated for a split training-test set.  If True, it will use
            a test set, if False it will use the entire <X> array for both
            training and testing.
        """

        self._check_regr("r_squared")
        if split is True:
            self._check_split("r_squared")
            Y_hat = self.predict(self._X_test)
            r_squared = 1 - (np.sum((self._Y_test - Y_hat)**2))/\
                            (np.sum((self._Y_test - np.mean(self._Y_test))**2))
        else:
            Y_hat = self.predict(self._X)
            r_squared = 1 - (np.sum((self._Y - Y_hat)**2))/\
                            (np.sum((self._Y - np.mean(self._Y))**2))

        return r_squared

    def k_fold(self, k, degree, sigma = None, alpha = None):
        """
            ---PURPOSE------------------------------------

            Implements k-fold cross-validation

            ---INPUT--------------------------------------

            k               Integer greater than 1
            degree          Integer greater than zero, or array of integers
                            greater than zero

            ---OPTIONAL-INPUT-----------------------------

            sigma           Scalar value
            alpha           Real number greater than zero

            ---OUTPUT-------------------------------------

            R2              Float
            MSE             Float
            variance        1-D array

            ---NOTES--------------------------------------

            If the dataset has been split, will apply k-fold to the training
            set.  If not, it will apply it to the entire dataset.

            The parameter <sigma> is representative of the standard deviation
            in a Gaussian noise that is expected to exist in the dataset.  It
            is assumed that the dataset output is a function of the form:

                        y = f(x_1, x_2, ..., x_p) + N(0, sigma)

            Where N(mu, sigma) is a normal distribution with a mean value of mu
            and a standard deviation of sigma.
        """

        try:
            if degree == int(degree) and degree > 0:
                degree = int(degree)
        except ValueError:
            error_msg = (f"\n\nParameter <degree> in <Regression.k_fold> "
                         f"must be an integer greater than zero\n\t"
                         f"type(degree) = {type(degree)}")
            raise ValueError(error_msg)

        if alpha is not None:
            try:
                if alpha > 0:
                    alpha = float(alpha)
                elif alpha <= 0:
                    raise ValueError()
            except TypeError:
                error_msg = (f"\n\nParameter <alpha> in <Regression.k_fold> "
                             f"must be a number greater than zero\n\t"
                             f"type(alpha) = {type(alpha)}")
                raise TypeError(error_msg)
            except ValueError:
                error_msg = (f"\n\nParameter <alpha> in <Regression.k_fold> "
                             f"must be a number greater than zero\n\t"
                             f"alpha = {alpha} --> suggest changing to: alpha "
                             f"= 1E-5")
                raise ValueError(error_msg)

        error_msg = (f"\n\nParameter <k> in method <Regression.k_fold> must be"
                     f" an integer greater than one\n\t")

        if not isinstance(k, int):
            error_msg += f"type(k) = {type(k)}"
            raise TypeError(error_msg)
        elif k < 2:
            error_msg += f"k = {k}"
            if self._N >= 500:
                error_msg += " --> suggest changing to: k = 10"
            raise ValueError(error_msg)

        N_subsample = self._X.shape[0]//k
        N_remain = self._X.shape[0]%k
        N_tot = self._X.shape[0] - N_remain

        if N_subsample < 10:
            warning_msg = (f"Implementing k-fold cross-validation in method "
                           f"<Regression.k_fold> with k = {k} leads to "
                           f"subsamples of size {N_subsample}, which are small"
                           f", and may thus lead to misleading results.")
            warnings.warn(warning_msg)
        elif N_subsample <= 1:
            error_msg = (f"Implementing k-fold cross-validation in method "
                           f"<Regression.k_fold> with k = {k} leads to "
                           f"subsamples of size {N_subsample}.\n\nSubsamples "
                           f"must contain more than one element.")

            raise ValueError(error_msg)

        idx_shuffle = np.random.permutation(N_tot)
        X_shuffle = self._X[idx_shuffle]
        Y_shuffle = self._Y[idx_shuffle]
        X_split = X_shuffle.reshape((k, N_subsample, self._p))
        Y_split = Y_shuffle.reshape((k, N_subsample))

        MSE = np.zeros(k)
        R2 = np.zeros(k)
        variance = []

        for i in range(k):
            X_train = np.delete(X_split, i, axis = 0)
            X_train = X_train.reshape(((k-1)*N_subsample, 2))
            X_test = X_split[i]

            Y_train = np.delete(Y_split, i, axis = 0)
            Y_train = Y_train.reshape((k-1)*N_subsample)
            Y_test = Y_split[i]

            beta, var, exponents = \
            self._internal_poly(X_train, Y_train, degree, "k_fold", alpha)

            variance.append(var)

            Y_hat = self._internal_predict(X_test, beta, exponents)

            R2[i] = 1 - (np.sum((Y_test - Y_hat)**2))/\
                            (np.sum((Y_test - np.mean(Y_test))**2))

            MSE[i] = np.mean((Y_test - Y_hat)**2)

        return np.mean(R2), np.mean(MSE)

    def plot(self, detail = 0.5, xlabel = None, ylabel = None, zlabel = None,
    savename = None, plot_points = True):
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
            plot_points     Boolean

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
        self._check_regr("plot")
        plt.style.use("seaborn")

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
            if plot_points is True:
                plt.plot(self._X[:,0], self._Y, "x", ms = 10)
            ymin, ymax = plt.ylim()
            plt.plot(x, F, "r-")
            plt.ylim(ymin, ymax)
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

            if plot_points is True:
                surf = ax.scatter(self._X[:,0], self._X[:,1],
                self._Y, s = 10, marker = ".", alpha = 0.5)

            if plot_points is True:
                zmin, zmax = ax.get_zlim()

            surf = ax.plot_surface(x, y, F, cmap=cm.terrain, alpha = 1,#0.5,
            antialiased=True, rcount = 116, ccount = 116)

            if plot_points is True:
                ax.set_zlim(zmin, zmax)

            if xlabel is not None:
                ax.set_xlabel("\n\n\n" + xlabel + "\n", linespacing = 3)
            if ylabel is not None:
                ax.set_ylabel("\n\n\n" + ylabel + "\n", linespacing = 3)
            if zlabel is not None:
                ax.set_zlabel("\n\n\n" + zlabel + "\n", linespacing = 3)

            figManager = plt.get_current_fig_manager()
            figManager.full_screen_toggle()

            if savename is not None:
                plt.savefig(savename, dpi = 250)
                plt.close()
            else:
                plt.show()

    def reset(self):
        """
            Resets the object to its initial state.
        """
        self.__init__(self._X_backup, self._Y_backup, dtype = self._dtype)

    def _design(self, X, degree, method):
        """
            ---PURPOSE------------------------------------

            Returns a design matrix for a multidimensional polynomial of the
            given <degree>, and their respective N-dimensional polynomial
            exponents, for a given set of features <X>

            ---INPUT--------------------------------------

            X           2-D Array of features
            degree      Integer greater than zero

            ---OUTPUT-------------------------------------

            A           2-D array
            exponents   1-D array


        """
        # Checking that <degree> is an integer greater than zero
        try:
            if degree == int(degree) and degree > 0:
                degree = int(degree)
        except ValueError:
            error_msg = (f"\n\nParameter <degree> in <Regression.{method}> "
                         f"must be an integer greater than zero\n\t"
                         f"type(degree) = {type(degree)}")
            raise ValueError(error_msg)

        M = int(degree) + 1

        # Setting up all the cross terms of the polynomial
        powers = np.arange(0, M, 1)
        powers = np.repeat(powers, self._p)
        exponents = list(permutations(powers, self._p))
        exponents = np.unique(exponents, axis = 0)

        # Excluding terms whose total is greater than <degree>
        if self._p != 1:
            expo_sum = np.sum(exponents, axis = 1)
            valid_idx = np.where(np.less_equal(expo_sum, degree))[0]
            exponents = np.array(exponents, dtype = np.int64)
            exponents = exponents[valid_idx]
        else:
            exponents = np.array(exponents, dtype = np.int64)

        # Creating the design matrix
        if self._p > 1:
            A = np.zeros((X.shape[0], exponents.shape[0]))
            for n,exponent in enumerate(exponents):
                A[:,n] = np.prod(X**exponent, axis = 1)
        else:
            A = np.zeros((X.shape[0], exponents.shape[0]))
            for n,exponent in enumerate(exponents):
                A[:,n] = X[:,0]**exponent

        return A, exponents

    def _check_regr(self, method_name):
        """
            Helper method to inform user that the object instance has not yet
            had regression applied to it in a situation where completion is
            necessary.
        """
        if self._complete is False:
            error_msg = (f"\n\nInvalid usage of <Regression> object; must "
                         f"first select a regression method before calling "
                         f"method <Regression.{method_name}>\n\nExample:\n\n"
                         f"IN[1]\tfoo = Regression([[1],[2],[3]], [3,5,8])\n\t"
                         f"foo.poly(degree = 2) # Selection of the regression"
                         f" method")
            raise Exception(error_msg)

    def _check_not_regr(self, method_name):
        """
            Helper method to inform user that the object instance has had
            regression applied to it in a situation where incompletion is
            necessary.
        """
        if self._complete is True:
            error_msg = (f"\n\nInvalid usage of <Regression> object; cannot "
                         f"apply method <Regression.{method_name}> after a "
                         f"regression has been implemented.\n\tCall method "
                         f"<Regression.reset> to return object to its original "
                         f"state.")
            raise Exception(error_msg)

    def _check_split(self, method_name):
        """
            Helper method to inform user that the object instance's dataset has
            not yet been split into training and testing data.
        """
        if self._split is False:
            error_msg = (f"\n\nInvalid usage of <Regression> object; must "
                         f"first split the dataset into training and testing "
                         f"data with <Regression.split> before calling the "
                         f"method <Regression.{method_name}>")
            raise Exception(error_msg)

    def _check_predict(self, method_name):
        """
            Helper method that returns the prediction of the original dataset,
            or training set (if it has been split)

            Raises an error if beta has not yet been calculated.
        """
        self._check_regr(method_name)
        if self._predicted is False:
            self._Y_hat = \
            self._internal_predict(self._X, self._beta, self._exponents)
            self._predicted = True

    def _internal_poly(self, X, y, degree, method, alpha = None):
        """
            ---PURPOSE------------------------------------

            Implements regression for a multidimensional polynomial of the
            given <degree> for the k-fold algorithm method.

            Can also implement ridge regression by passing a number greater
            than zero to parameter <alpha>.

            ---INPUT--------------------------------------

            X               2-D NumPy array of shape (N, p)
            y               1-D NumPy array of shape (N,)
            degree          Integer greater than zero
            method          String

            ---OPTIONAL-INPUT-----------------------------

            alpha           Real number greater than zero or None

            ---OUTPUT-------------------------------------

            beta
            variance
            exponents
        """

        A, exponents = self._design(X, degree, method)

        # Implementing the least-squares method
        if alpha is None:
            # If ridge regression is not implemented
            try:
                beta = np.linalg.inv(A.T @ A)
                variance = np.diagonal(beta)
                beta = beta @ A.T @ y
            except np.linalg.LinAlgError:
                error_msg = (f"\n\nThe design matrix product could not be "
                             f"inverted because it is a singular matrix.\n\n\t"
                             f"(XᵀX)⁻¹ = undefined\n\nTry setting parameter "
                             f"<alpha> in <Regression.k_fold> to 1E-4; this "
                             f"implements ridge regression, and may lead to an"
                             f" approximate solution.")

                raise np.linalg.LinAlgError(error_msg)
        else:
            # If ridge regression is implemented
            beta = A.T @ A
            try:
                beta = np.linalg.inv(beta + alpha*\
                np.identity(beta.shape[0]))
                variance = np.diagonal(beta)
                beta = beta @ A.T @ y
            except np.linalg.LinAlgError:
                error_msg = (f"\n\nThe design matrix product could not be "
                             f"inverted because it is a singular matrix.\n\n\t"
                             f"(XᵀX + {alpha:g}*I)⁻¹ = undefined\n\nTry "
                             f"increasing parameter <alpha> in method "
                             f"<Regression.k_fold> to find a non-singular "
                             f"matrix.")
                raise np.linalg.LinAlgError(error_msg)

        return beta, variance, exponents

    def _internal_predict(self, X, beta, exponents):
        """
            ---PURPOSE------------------------------------

            Assuming that a regression has taken place, will predict the output given
            by a set of inputs in <X>

            ---INPUT--------------------------------------

            X               Numerical array of shape (M, p)
            beta
            exponents

            ---OUTPUT-------------------------------------

            Y           Array of shape (M,)
        """
        if self._p > 1:
            A = np.zeros((X.shape[0], exponents.shape[0]))
            for n,exponent in enumerate(exponents):
                A[:,n] = np.prod(X**exponent, axis = 1)
        else:
            A = np.zeros((X.shape[0], exponents.shape[0]))
            for n,exponent in enumerate(exponents):
                A[:,n] = X[:,0]**exponent

        Y_hat = A @ beta

        return Y_hat

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
            factors = [f"{abs(b)}"]
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

    def __str__(self):
        self._check_regr("__str__")
        return self._readable
