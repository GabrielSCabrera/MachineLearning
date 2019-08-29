from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

N = 200
M = 50

X1 = np.random.normal(0, 6, (N//2, 2))
X2 = np.random.normal(13, 8, (N//2, 2))

Y1 = np.zeros(N//2)
Y2 = np.ones(N//2)

X = np.concatenate([X1, X2])
Y = np.concatenate([Y1, Y2])

beta = np.matmul(np.linalg.inv(np.matmul(X.T, X)),np.matmul(X.T, Y))

X_hat = np.random.normal(6, 10, (2, M))
Y_hat = np.matmul(X_hat.T, beta)
Y_hat[Y_hat < 0.5] = 0
Y_hat[Y_hat >= 0.5] = 1

for i,j,k in zip(X[:,0], X[:,1], Y):
    if k == 0:
        plt.plot(i,j,"ro", alpha = 0.2)
    else:
        plt.plot(i,j,"bo", alpha = 0.2)

for i,j,k in zip(X_hat[0], X_hat[1], Y_hat):
    if k == 0:
        plt.plot(i,j,"r^")
    else:
        plt.plot(i,j,"b^")

plt.show()
