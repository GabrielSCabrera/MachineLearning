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
        A = np.zeros((self._N, exponents.shape[0]))
        for n,exponent in enumerate(exponents):
            A[:,n] = np.prod(self._X**exponent, axis = 1)
    else:
        A = np.zeros((self._N, exponents.shape[0]))
        for n,exponent in enumerate(exponents):
            A[:,n] = self._X[:,0]**exponent

    clf = skl.Lasso(alpha=alpha)
    clf.fit(A, self._Y)
    beta = clf.coef_
    self._beta = beta
    self._beta[0] = clf.intercept_

    self._predicted = True
    self._readable, self._terms = self._poly_str(exponents, self._beta)
    self._complete = True
    self._exponents = exponents
