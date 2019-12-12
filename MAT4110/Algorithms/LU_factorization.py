import numpy as np

def LU_decompose(A):
    """
        Parameter 'A' should be a numerical numpy array with 'A.ndim == 2' of
        shape (N,N).  It should only contain real numbers.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("LU Decomposition only valid for an NxN matrix!")

    # Matrix Size
    N = A.shape[0]

    A0 = A.copy()

    # Initializing Matrices L and U
    L,U = np.zeros((N,N)), np.zeros((N,N))

    # Initial Conditions
    L[:,0] = A[:,0]/A[0,0]
    U[0] = A[0]
    for i in range(1, N):
        A = A - L[:,i-1,None] @ U[i-1,None]
        L[:,i] = A[:,i]/A[i,i]
        U[i] = A[i]

    print(f"(1/3){L*3}\n(1/3){U*3}")

A = np.array([[3,4],[5,6]])
LU_decompose(A)
