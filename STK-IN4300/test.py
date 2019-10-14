from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

X1 = np.array([20, 60, 53]) # Ages
X2 = np.array([180, 195, 210]) # Heights
Y = np.array([0, 1, 0]) # Basketball or Baseball

X = np.ones((3,3))
X[:,1] = X1
X[:,2] = X2
print(X)

beta = np.matmul(np.linalg.inv(np.matmul(X.T, X)),np.matmul(X.T, Y))
X1_hat = np.array([15, 30, 55]) # Ages
X2_hat = np.array([165, 180, 200]) # Heights

X_hat = np.ones((3,3))
X_hat[1] = X1_hat
X_hat[2] = X2_hat

Y_hat = np.matmul(X_hat.T, beta)
print(Y_hat)
Y_hat[Y_hat < 0.5] = 0
Y_hat[Y_hat >= 0.5] = 1

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X1, X2, Y, c = "b")
ax.scatter(X1_hat, X2_hat, Y_hat, c = "r")
plt.show()
