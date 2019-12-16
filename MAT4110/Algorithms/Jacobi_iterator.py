import numpy as np

def jacobi_iterator(A, b, k, x0 = None, seed = None):
    """
        Parameter 'A' should be a numerical numpy array with 'A.ndim == 2' of
        shape (N,N).  It should only contain real numbers.

        Parameters 'b', 'x0' should be a numerical numpy array with
        'A.ndim == 1' of shape (N,). They should only contain real numbers.

        Parameter 'k' is the number of iterations to be taken, and should be
        and integer

        Solves the equation 'Ax = b' and returns a vector 'x' of length N
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(jacobi_iterator.__doc__)

    # Matrix Size
    N = A.shape[0]

    if b.ndim != 1 or b.shape[0] != N:
        raise ValueError(jacobi_iterator.__doc__)

    # If no guess 'x0' is given, randomly generates an initial guess
    if x0 is None:
        # If random seed is given, initializes it
        if seed is not None:
            np.random.seed(seed)
        x0 = np.random.random(N)
    else:
        if x0.ndim != 1 or x0.shape[0] != N:
            raise ValueError(jacobi_iterator.__doc__)

    # Initializing Matrices D and B
    D = np.diag(np.diagonal(A))
    B = A - D

    # Inverting the diagonal
    D = np.diag(1/np.diagonal(D))

    # Creating the initial solution 'x'
    x = x0[:,None].copy()

    # Making 'b' a column vector
    b = b[:,None]

    # Algorithm
    for i in range(k):
        x = D @ (-B @ x + b)

    return x.squeeze()

if __name__ == "__main__":
    A = np.array([[1,0],[2,3]])
    b = np.array([1,2])
    k = 10
    seed = 1
    x = jacobi_iterator(A, b, k, seed = seed)
    print(f"Ax == b is {np.all(np.isclose(A @ x[:,None], b[:,None]))}")
