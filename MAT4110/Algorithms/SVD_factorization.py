import numpy as np

def SVD_decompose(A):
    """
        Parameter 'A' should be a numerical numpy array with 'A.ndim == 2' of
        shape (M,N).  It should only contain real numbers.
    """
    if A.ndim != 2:
        raise ValueError(SVD_decompose.__doc__)

    # Matrix Size
    M,N = A.shape[0], A.shape[1]

    # Matrix Rank
    k = np.min([np.linalg.matrix_rank(A), np.min([M,N])])

    # Initializing Matrices U, S, and V
    U,S,V = np.zeros((M,k)), np.zeros((k,k)), np.zeros((N,k))

    # Eigenvalues/Eigenvectors
    eigvals, eigvecs = np.linalg.eig(A.T @ A)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:,order]
    V = eigvecs

    S1 = np.diag(eigvals**0.5)[:k,:k]
    D_inv = np.diag(eigvals**(-0.5))[:k,:k]

    V1 = V[:,:k]

    U1 = A @ V1 @ D_inv

    return U1,S1,V1

if __name__ == "__main__":
    A = np.array([[3,2,2],[2,3,-2]])
    U,S,V = SVD_decompose(A)
    print(f"UΣVᵀ == A is {np.all(np.isclose(U @ S @ V.T, A))}")
