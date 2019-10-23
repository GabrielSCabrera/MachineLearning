import numpy as np

a1 = np.array([[1],[2]])
b1 = np.array([[1],[2],[3]])

a = np.array([[[1],[2]], [[1],[2]], [[1],[2]]])
b = np.array([[[1],[2],[3]], [[1],[2],[3]], [[1],[2],[3]]])
c = np.einsum("ijk,ikj->ijk",a,b)
print(c)
print(np.outer(a1, b1))
