import numpy as np

def LU_decompose(A):
    """
        Parameter 'A' should be a numerical numpy array with 'A.ndim == 2' of
        shape (N,N).  It should only contain real numbers.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(LU_decompose.__doc__)

    # Matrix Size
    N = A.shape[0]

    # Initializing Matrices L and U
    L,U = np.zeros((N,N)), np.zeros((N,N))

    # Algorithm
    for i in range(N):
        L[:,i] = A[:,i]/A[i,i]
        U[i] = A[i]
        A = A - L[:,i,None] @ U[i,None]

    return L, U

if __name__ == "__main__":
    A = np.array([[3,4],[5,6]])
    L,U = LU_decompose(A)
    print(f"LU == A is {np.all(np.isclose(L @ U, A))}")
