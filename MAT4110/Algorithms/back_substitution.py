import numpy as np

def back_substitution(A, b):
    """
        Parameter 'A' should be a numerical numpy array with 'A.ndim == 2' of
        shape (N,N).  It should only contain real numbers.

        Parameter 'b' should be a numerical numpy array with 'A.ndim == 1' of
        shape (N,). It should only contain real numbers.

        Solves the equation 'Ax = b' and returns a vector 'x' of length N

        'A' must be an upper triangular matrix!
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1] or np.any(np.triu(A) != A):
        raise ValueError(back_substitution.__doc__)

    # Matrix Size
    N = A.shape[0]

    if b.ndim != 1 or b.shape[0] != N:
        raise ValueError(back_substitution.__doc__)

    # Initialize empty array of solutions
    x = np.zeros(N)

    # Algorithm
    for k in range(N):
        i = N - k - 1
        x[i] = (b[i] + np.sum(-A[i,i+1:]*x[i+1:]))/A[i,i]

    return x

if __name__ == "__main__":
    A = np.array([[1,2],[0,3]])
    b = np.array([1,2])
    x = back_substitution(A, b)
    print(A @ x[:,None])
