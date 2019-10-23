from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations
from multiprocessing import Pool
import matplotlib.pyplot as plt
from imageio import imread
from matplotlib import cm
import numpy as np

class Poly():

    """CONSTRUCTOR"""

    def __init__(self, X, Y, dtype = np.float64):
        """
            PURPOSE
                Initializes an instance of Poly() without implementing any
                train-test splitting, or regression methods.

            INPUT
                X       2-D Array of real numbers (N, p)
                        OR  1-D Array of real numbers (N,)

                Y       2-D Array of real numbers (N, q)
                        OR  1-D Array of real numbers (N,)

            OPTIONAL INPUT
                dtype   Datatype - see NumPy's documentation for dtype

            NOTES
                N represents the number of inputs
                p represents the number of features
                q represents the number of labels
        """

        self._X = X                         # TODO: Check dims, element types
        self._Y = Y                         # TODO: Check dims, element types
        self._N = self._X.shape[0]
        if self._X.ndim == 1:
            self._p = 1
            self._X = self._X[:,np.newaxis]
        else:
            self._p = self._X.shape[1]

        if self._Y.ndim == 1:
            self._q = 1
            self._Y = self._Y[:,np.newaxis]
        else:
            self._q = self._Y.shape[1]

        # Metadata
        self._is_split = False
        self._is_regr = False

    """FRONTEND METHODS"""

    def cross_validate(self, k, d_min, d_max, a_min, a_max, a_N):
        """
            PURPOSE
                Implements k-fold cross validation on the complete dataset by
                calculating the MSE and R²-score for a variety of polynomial
                complexities and ridge regression hyperparameters

            INPUT
                k                   Integer
                d_min               Integer
                d_max               Integer
                a_min               Real Number
                a_max               Real Number
                a_N                 Integer

            OUTPUT
                D                   2-D Array
                A                   2-D Array
                MSE                 2-D Array
                R2                  2-D Array
        """
        pool = Pool()
        a_array = np.linspace(a_min, a_max, a_N)
        d_array = np.arange(d_min, d_max+1)
        d_N = len(d_array)
        A,D = np.meshgrid(a_array, d_array)
        MSE = np.zeros((d_max - d_min + 1, a_N))
        R2 = np.zeros_like(MSE)
        for m,d in enumerate(d_array[::-1]):
            X, exponents = self.design(degree = d)
            for n,a in enumerate(a_array):
                perc = 100*(n + m*a_N)/(d_N*a_N)
                print(f"\r{perc:>5.1f}%", end = "")
                gen_args = self._generate_cv_sets(X, self._Y, k, a)
                step = np.array(pool.starmap(self._validate_ridge, gen_args))
                MSE[d_N - m - 1,n], R2[d_N - m - 1,n] = np.mean(step, axis = 0)
        print()
        return D, A, MSE, R2

    def design(self, degree, set = "all"):
        """
            PURPOSE
                Returns a design matrix with the input X, as well as the
                corresponding terms' exponents.

            INPUTS
                degree          Integer

            OPTIONAL INPUTS
                set             String in ["all", "test", "train"]

            OUTPUT
                A               2-D NumPy Array
                exponents       1-D or 2-D Array of integers

            NOTES
                Parameter <degree> represents the polynomial degree for which
                we will later calculate the vector of coefficients: beta

                If the parameter <set> is "all", the design matrix will contain
                the entire dataset.  If <set> is "train", it will contain the
                training set, and if <set> is "test", it will contain the
                testing set.

                If <set = "train"> or <set = "test">, then the method
                <Poly.split> must have already been executed or an exception
                is raised.
        """
        # TODO: Check that "set" is one of the given options, or raise error
        X,Y = self._get_XY(set, "design")
        A, exponents = self._design(X, degree, method = "design")
        self._is_designed = True
        return A, exponents

    def OLS(self, degree, set = "all"):
        """
            PURPOSE
                Implements ordinary least-squares regression on the dataset or
                selected subset.

            INPUT
                degree          Integer

            OPTIONAL INPUT
                set             String in ["all", "test", "train"]
        """
        X,Y = self._get_XY(set = set, method = "OLS")
        A, self._exponents = self.design(degree = degree, set = set)
        self._beta = self._OLS(A, Y)
        self._is_regr = True

    def ridge(self, degree, alpha, set = "all"):
        """
            PURPOSE
                Implements ridge regression on the dataset or selected subset.
            INPUT
                degree          Integer
                alpha           Real Number

            OPTIONAL INPUT
                set             String in ["all", "test", "train"]
        """
        X,Y = self._get_XY(set = set, method = "OLS")
        A, self._exponents = self.design(degree = degree, set = set)
        self._beta = self._ridge(A, Y, alpha)
        self._is_regr = True

    def split(self, test_ratio = 0.25):
        """
            PURPOSE
                Splits the dataset into a training set and testing set

            OPTIONAL INPUT
                test_ratio      Float in range (0,1)

            OUTPUT
                X_train         2-D Array
                X_test          2-D Array
                Y_train         2-D Array
                Y_test          2-D Array

            NOTES
                Parameter <test_ratio> represents what portion of the dataset
                should become part of the testing set on a scale of 0 to 1,
                not including 0 and 1 themselves.
        """
        self._X_train, self._X_test, self._Y_train, self._Y_test = \
        self._split(self._X, self._Y, test_ratio)
        self._is_split = True

        return self._X_train, self._X_test, self._Y_train, self._Y_test

    """BACKEND METHODS"""

    def _check_regr(self, method):
        """
            PURPOSE
                Helper method to inform user that the object instance has not
                yet had regression applied to it in a situation where
                completion is necessary.

            INPUT
                method          String
        """
        if self._is_regr is False:
            error_msg = (f"\n\nInvalid usage of <Poly> object; must "
                         f"first select a regression method before calling "
                         f"method <Poly.{method}>\n\nExample:\n\n"
                         f"IN[1]\tfoo = Poly([[1],[2],[3]], [3,5,8])\n\t"
                         f"foo.poly(degree = 2) # Selection of a regression"
                         f" method")
            raise Exception(error_msg)

    def _check_split(self, method):
        """
            PURPOSE
                Helper method to inform user that the object instance has not
                yet been split into training and testing data.

            INPUT
                method          String
        """
        if self._is_split is False:
            error_msg = (f"\n\nInvalid usage of <Poly> object; must "
                         f"first split the dataset into training and testing "
                         f"data with <Poly.split> before calling "
                         f"<Regression.{method}>")
            raise Exception(error_msg)

    def _design(self, X, degree, method):
        """
            PURPOSE
                Backend method to create a design matrix, given a set of inputs

            INPUT
                method          String

            OUTPUT
                A               2-D NumPy Array
                exponents       1-D or 2-D Array of integers
        """

        # Making sure we don't exceed the size of an unsigned 8-bit integer
        if degree > 256:
            msg = f"Maximum polynomial degree exceeded in method <R.{method}>"
            raise ValueError(msg)

        N = X.shape[0]
        p = X.shape[1]

        # Setting up all the cross terms of the polynomial
        powers = np.arange(0, degree + 1, 1)
        powers = np.repeat(powers, p)
        exponents = list(permutations(powers, p))
        exponents = np.unique(exponents, axis = 0)

        # Excluding terms whose total is greater than <degree>
        if p != 1:
            expo_sum = np.sum(exponents, axis = 1)
            valid_idx = np.where(np.less_equal(expo_sum, degree))[0]
            exponents = np.array(exponents, dtype = np.uint16)
            exponents = exponents[valid_idx]
        else:
            exponents = np.array(exponents, dtype = np.uint16)

        if p > 1:
            A = np.zeros((N, exponents.shape[0]))
            for n,exponent in enumerate(exponents):
                A[:,n] = np.prod(X**exponent, axis = 1)
        else:
            A = np.zeros((N, exponents.shape[0]))
            for n,exponent in enumerate(exponents):
                A[:,n] = X[:,0]**exponent

        return A, exponents

    def _generate_cv_sets(self, X, Y, k, alpha, f_xy = None):
        """
            PURPOSE
                A generator method that iteratively yields each set of inputs
                for the multiprocessing starmap in method Poly.cross_validate()

            INPUT
                X               2-D Array
                Y               2-D Array
                k               Integer
                alpha           Real Number

            OPTIONAL INPUT
                f_xy        # TODO: NOT YET IMPLEMENTED

            OUTPUT
                X_train, X_test, Y_train, Y_test, alpha
        """
        # TODO: Implement f_xy here and in self._validate, self.cross_validate
        N = X.shape[0]
        N_subsample = N//k
        N_remain = N%k
        N_tot = N - N_remain
        p = X.shape[1]

        idx_shuffle = np.random.permutation(N_tot)
        X_shuffle = X[idx_shuffle]
        Y_shuffle = Y[idx_shuffle]
        X_split = X_shuffle.reshape((k, N_subsample, p))
        Y_split = Y_shuffle.reshape((k, N_subsample))

        if f_xy is not None:
            f_shuffle = f_xy[idx_shuffle]
            f_split = f_shuffle.reshape((k, N_subsample))

        for i in range(k):
            X_test = X_split[i].copy()
            X_train = np.delete(X_split, i, axis = 0)
            X_train = X_train.reshape(((k-1)*N_subsample, p))

            if f_xy is not None:
                f_test = f_split[i].copy()
                f_train = np.delete(f_split, i, axis = 0)
                f_train = f_train.reshape((k-1)*N_subsample)

            Y_test = Y_split[i].copy()
            Y_train = np.delete(Y_split, i, axis = 0)
            Y_train = Y_train.reshape((k-1)*N_subsample)

            yield X_train, X_test, Y_train, Y_test, alpha

    def _get_XY(self, set, method):
        """
            PURPOSE
                Extracts the desired subset of the input data based on the
                parameter "set"

            INPUT
                set             String in ["all", "test", "train"]
                method          String

            OUTPUT
                X               2-D Array
                Y               2-D Array
        """
        if set == "all":
            X = self._X
            Y = self._Y
        elif set == "train":
            _check_split(method)
            X = self._X_train
            Y = self._Y_train
        elif set == "test":
            _check_split(method)
            X = self._X_test
            Y = self._Y_test
        return X,Y

    def _OLS(self, A, Y):
        """
            PURPOSE
                Implements ordinary least-squares regression on the given
                design matrix A and output Y

            INPUT
                A               2-D Array
                Y               2-D Array

            OUTPUT
                beta            1-D or 2-D Array
        """
        beta = A.T @ A
        try:
            beta = np.linalg.inv(beta)
        except np.linalg.LinAlgError:
            error_msg = (f"\n\nMatrix could not be inverted because it is a"
                         f"singular matrix.\n\n\t(XᵀX + {alpha:g}*I)⁻¹ = "
                         f"undefined\n\nTry implementing \033[3mRidge "
                         f"Regression\033[m with the method <Poly.ridge>.")
            raise np.linalg.LinAlgError(error_msg)
        beta = beta @ A.T @ Y
        return beta

    def _poly_str(self, exponents, beta):
        """
            PURPOSE
                Takes a set of exponents and a vector of coefficients and uses
                these to create a list of strings which are human-readable.

                Should not be accessed by users.

            INPUT
                exponents       2-D Array
                beta            1-D Array

            OUTPUT
                readable        String containing human-readable polynomial
                terms           List of evaluatable strings of length N

            NOTES
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

    def _ridge(self, A, Y, alpha):
        """
            PURPOSE
                Implements ridge regression on the given design matrix A and
                output Y

            INPUT
                A               2-D Array
                Y               2-D Array

            OUTPUT
                beta            1-D or 2-D Array
        """
        beta = A.T @ A + alpha*np.identity(A.shape[1])
        try:
            beta = np.linalg.inv(beta)
        except np.linalg.LinAlgError:
            error_msg = (f"\n\nMatrix could not be inverted because it is a "
                         f"\033[3msingular matrix\033[m\n\n\t"
                         f"(XᵀX + {alpha:g}*I)⁻¹ = undefined\n\nTry tweaking "
                         f"parameter \033[1malpha\033[m in <Poly.ridge>.")
            raise np.linalg.LinAlgError(error_msg)
        beta = beta @ A.T @ Y
        return beta

    def _ridge_SVD(self, A, Y, alpha):
        """
            PURPOSE
                Implements ridge regression on the given design matrix A and
                output Y, via the singular-value decomposition

            INPUT
                A               2-D Array
                Y               2-D Array

            OUTPUT
                beta            1-D or 2-D Array
        """
        U, S_diag, V = np.linalg.svd(A, full_matrices = False)
        V = V.T
        S = np.diag(S_diag/(S_diag**2 - alpha))
        beta = V @ S @ U.T @ Y
        return beta

    def _split(self, X, Y, test_ratio):
        """
            PURPOSE
                Splits the dataset into a training set and testing set

            INPUT
                X               2-D Array
                Y               2-D Array

            OPTIONAL INPUT
                test_ratio      Float in range (0,1)

            OUTPUT
                X_train         2-D Array
                X_test          2-D Array
                Y_train         2-D Array
                Y_test          2-D Array

            NOTES
                Parameter <test_ratio> represents what portion of the dataset
                should become part of the testing set on a scale of 0 to 1,
                not including 0 and 1 themselves.
        """
        N = X.shape[0]
        N_test = int(round(N*test_ratio))
        test_idx = np.random.choice(a = N, size = N_test, replace = False)
        train_idx = np.delete(np.arange(0,N,1), test_idx)
        X_test = X[test_idx]
        Y_test = Y[test_idx]
        X_train = np.delete(X, test_idx, axis = 0)
        Y_train = np.delete(Y, test_idx)

        N_train = X_train.shape[0]
        N_test = X_test.shape[0]

        return X_train, X_test, Y_train, Y_test

    def _validate_ridge(self, X_train, X_test, Y_train, Y_test, alpha):
        """
            PURPOSE
                Returns the mean k-fold cross-validation data for the MSE and
                R²-score, for a given set of training and testing datasets.

            INPUT
                X_train             2-D Array
                X_test              2-D Array
                Y_train             2-D Array
                Y_test              2-D Array
                alpha               Real Number

            OUTPUT
                MSE                 Float
                R2                  Float
        """
        beta = self._ridge_SVD(X_train, Y_train, alpha)
        Y_hat = X_test @ beta

        MSE = np.mean((Y_test - Y_hat)**2)
        R2 = 1 - (np.sum((Y_test - Y_hat)**2))/\
                 (np.sum((Y_test - np.mean(Y_test))**2))

        return MSE, R2

    """OBJECT PROPERTIES"""

    @property
    def beta(self):
        """
            PURPOSE
                Returns the vector of coefficients

            OUTPUT
                beta        1-D or 2-D NumPy Array
        """
        self._check_regr("beta")
        return self._beta

    @property
    def N(self):
        """
            PURPOSE
                Returns the number of inputs N in matrices X and Y

            OUTPUT
                N           Integer value
        """
        return self._N

    @property
    def N_train(self):
        """
            PURPOSE
                Returns the number of inputs N in matrices X_train and Y_train

            OUTPUT
                N_train     Integer value
        """
        self._check_split("N_train")
        return self._N_train

    @property
    def N_test(self):
        """
            PURPOSE
                Returns the number of inputs N in matrices X_test and Y_test

            OUTPUT
                N_test      Integer value
        """
        self._check_split("N_test")
        return self._N_test

    @property
    def p(self):
        """
            PURPOSE
                Returns the dimension p of the matrix X

            OUTPUT
                p           Integer value

            NOTES
                Dimension p – the number of features in our dataset
        """
        return self._p

    @property
    def q(self):
        """
            PURPOSE
                Returns the dimension q of the matrix Y

            OUTPUT
                q           Integer value

            NOTES
                Dimension q – the number of labels in our dataset
        """
        return self._q

    @property
    def terms(self):
        """
            PURPOSE
                Returns the terms corresponding to the columns of the currently
                implemented regression Poly.

            OUTPUT
                terms       1-D or 2-D NumPy Array
        """
        self._check_regr("terms")
        return self._terms

    @property
    def X(self):
        """
            PURPOSE
                Returns the set of features

            OUTPUT
                X           2-D Array
        """
        return self._X

    @property
    def X_test(self):
        """
            PURPOSE
                Returns the testing set of features; if Method.split() has not
                been called previously, will raise an Exception.

            OUTPUT
                X_test      2-D Array
        """
        self._check_split("X_test")
        return self._X_test

    @property
    def X_train(self):
        """
            PURPOSE
                Returns the training set of features; if Method.split() has not
                been called previously, will raise an Exception.

            OUTPUT
                X_train     2-D Array
        """
        self._check_split("X_train")
        return self._X_train

    @property
    def Y(self):
        """
            PURPOSE
                Returns the set of labels

            OUTPUT
                Y           2-D Array
        """
        return self._Y

    @property
    def Y_test(self):
        """
            PURPOSE
                Returns the testing set of labels; if Method.split() has not
                been called previously, will raise an Exception.

            OUTPUT
                Y_test      2-D Array
        """
        self._check_split("Y_test")
        return self._Y_test

    @property
    def Y_train(self):
        """
            PURPOSE
                Returns the training set of labels; if Method.split() has not
                been called previously, will raise an Exception.

            OUTPUT
                Y_train       2-D Array
        """
        self._check_split("Y_train")
        return self._Y_train

if __name__ == "__main__":

    def generate_Franke_data(x_min, x_max, N):

        # Generating NxN meshgrid of x,y values in range [x_min, x_max]
        x = np.random.random((N,N))*(x_max-x_min) + x_min
        y = np.random.random((N,N))*(x_max-x_min) + x_min

        # Calculating the values of the Franke function at each (x,y) coordinate
        Z = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        Z += 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        Z += 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        Z += -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        init_error = np.random.normal(0, 1, Z.shape)

        f_xy = Z.copy().flatten()
        Z = Z + init_error

        # Making compatible input arrays for Regression object
        X = np.zeros((x.shape[0]*x.shape[1], 2))
        X[:,0] = x.flatten()
        X[:,1] = y.flatten()
        Y = Z.flatten()

        return X, Y, f_xy

    def part_F():
        skip_real = 2
        # Importing the data
        terrain_data = imread("file.tif")
        # Resizing the data
        terrain_data = terrain_data[::skip_real,::skip_real]
        # Normalizing the data
        terrain_data = (terrain_data - np.mean(terrain_data))\
                       / np.std(terrain_data)

        dims = terrain_data.shape
        x, y = np.arange(0, dims[0]), np.arange(0, dims[1])
        X, Y = np.meshgrid(x,y)
        X_regr = np.array([X.flatten(), Y.flatten()]).T
        Y_regr = terrain_data.flatten()

        return X_regr, Y_regr

    X, Y, f_xy = generate_Franke_data(0, 1, 100)
    # X,Y = part_F()

    k = 10
    d_min = 2
    d_max = 10
    a_min = 0
    a_max = 1
    a_N = 50

    M = Poly(X, Y)

    D, A, MSE, R2 = M.cross_validate(k, d_min, d_max, a_min, a_max, a_N)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    fig.set_size_inches(8, 6)

    idx_min = np.unravel_index(MSE.argmin(), A.shape)
    print(f"d_min = {D[idx_min]:g}, a_min = {A[idx_min]:g}, MSE = {MSE[idx_min]}")
    ax.plot_surface(D, A, np.log(MSE), cmap = cm.magma)
    plt.show()
