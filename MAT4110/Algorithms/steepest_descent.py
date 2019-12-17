from forward_substitution import forward_substitution as sub
import numpy as np

def steepest_descent(A, b, k, x0 = None, seed = None, tol = 1E-10):
    """
        Parameter 'A' should be a numerical numpy array with 'A.ndim == 2' of
        shape (N,N).  It should be positive definite and symmetric.

        Parameters 'b', 'x0' should be a numerical numpy array with
        'A.ndim == 1' of shape (N,). They should only contain real numbers.

        Parameter 'k' is the number of iterations to be taken, and should be
        and integer

        Solves the equation 'Ax = b' and returns a vector 'x' of length N
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(steepest_descent.__doc__)

    # Matrix Size
    N = A.shape[0]

    symmetric = np.array_equal(A, A.T)
    positive_definite = np.all(np.greater(np.linalg.eigvals(A), 0))

    if b.ndim != 1 or b.shape[0] != N or not symmetric or not positive_definite:
        raise ValueError(steepest_descent.__doc__)

    # If no guess 'x0' is given, randomly generates an initial guess
    if x0 is None:
        # If random seed is given, initializes it
        if seed is not None:
            np.random.seed(seed)
        x0 = np.random.random(N)
    else:
        if x0.ndim != 1 or x0.shape[0] != N:
            raise ValueError(steepest_descent.__doc__)

    # Integrated Function
    def F(A, b, x):
        return 0.5*x.T @ A @ x - x.T @ b

    omega = 0
    b = b[:,None]
    x = x0.copy()[:,None]
    g = F(A, b, x)

    # Algorithm
    for i in range(k):
        # Check if norm of gradient is sufficiently small
        if np.linalg.norm(A @ x - b) < tol:
            break
        g = F(A, b, x - g)
        print(g.shape, A.shape)
        omega = (g.T @ g)/(g.T @ A @ g)
        x = x - omega*g

    return x.squeeze()

if __name__ == "__main__":
    A = np.array([[1,0],[0,1]])
    b = np.array([1,2])
    k = 10
    seed = 1
    x = steepest_descent(A, b, k, seed = seed)
    print(f"Ax == b is {np.all(np.isclose(A @ x[:,None], b[:,None]))}")
