import numpy as np

def cholesky_decompose(A):
    """
        Parameter 'A' should be a numerical numpy array with 'A.ndim == 2' of
        shape (N,N).  It should only contain real numbers.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(cholesky_decompose.__doc__)

    # Matrix Size
    N = A.shape[0]

    # Initializing Matrices L and D
    L, D = np.zeros((N,N)), np.zeros((N,N))

    # Algorithm
    for i in range(N):
        L[:,i] = A[:,i]/A[i,i]
        D[i,i] = A[i,i]
        A = A - D[i,i]*(L[:,i,None] @ L[:,i,None].T)

    return L, D

if __name__ == "__main__":
    A = np.array([[3,4],[4,6]])
    L,D = cholesky_decompose(A)
    print(f"LDLᵀ == A is {np.all(np.isclose(L @ D @ L.T, A))}")
