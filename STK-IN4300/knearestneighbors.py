import matplotlib.pyplot as plt
import numpy as np

k = 8
N = 200
M = 500

X1 = np.random.normal(0, 6, (N//2, 2))
X2 = np.random.normal(5, 8, (N//2, 2))

Y1 = np.zeros(N//2)
Y2 = np.ones(N//2)

X = np.concatenate([X1, X2])
Y = np.concatenate([Y1, Y2])

X_hat = np.random.normal(6, 10, (2, M))
Y_hat = np.zeros(M)

for n,i in enumerate(X_hat.T):
    diffs = np.linalg.norm(i - X, axis = 1)
    order = np.argsort(diffs)[:k]
    avg = np.round(np.mean(Y[order]))
    Y_hat[n] = avg

for i,j,k in zip(X[:,0], X[:,1], Y):
    if k == 0:
        plt.plot(i,j,"ro", alpha = 0.15)
    else:
        plt.plot(i,j,"bo", alpha = 0.15)

for i,j,k in zip(X_hat[0], X_hat[1], Y_hat):
    if k == 0:
        plt.plot(i,j,"r^")
    else:
        plt.plot(i,j,"b^")

plt.show()
