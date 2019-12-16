import numpy as np

def QR_decompose(A):
    """
        Parameter 'A' should be a numerical numpy array with 'A.ndim == 2' of
        shape (M,N) where M >= N.  It should only contain real numbers.
    """
    if A.ndim != 2 or A.shape[0] < A.shape[1]:
        raise ValueError(QR_decompose.__doc__)

    # Matrix Size
    M,N = A.shape[0], A.shape[1]

    # Initializing Matrices Q and R
    Q, R = np.zeros((M,M)), np.zeros((M,N))

    # Gram-Schmidt Algorithm
    for i in range(N):
        tot = np.zeros_like(A[:,i], dtype = np.float64)
        for j in range(i-1):
            tot += np.inner(Q[:,j], A[:,i]) * Q[:,j]
        w = A[:,i] - tot
        w_norm = np.linalg.norm(w)
        Q[:,i] = w/w_norm
        for j in range(i-1):
            R[j,i] = np.inner(Q[:,j], A[:,i])
        R[i,i] = w_norm

    return Q, R

if __name__ == "__main__":
    A = np.array([[2,1,-3],[0,0,-1],[0,1,4]])
    Q,R = QR_decompose(A)
    print(f"QR == A is {np.all(np.isclose(Q @ R, A))}")
