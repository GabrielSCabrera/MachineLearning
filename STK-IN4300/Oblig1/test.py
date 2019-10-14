import numpy as np
np.random.seed(69420666)

N = 5
p = 2
e_min = -1E2
e_max = 1E2

X = np.random.randint(e_min, e_max, (N,p))
w = np.random.randint(e_min, e_max, (p,1))
a = np.random.randint(e_min, e_max, (1,N))
b = np.random.randint(e_min, e_max, (1,N))

# w.T @ X           -->     (1,N)
# a + w.T @ X.T     -->     (1,N)
# b @ b.T           -->     S_i (b[:,i][0]**2)

tot = 0
for i in range(N):
    x_i = X[i,:,np.newaxis]
    tot += (b[:,i]**2)*((a[:,i] + w.T @ x_i)**2)

diag = np.zeros((N,N))
for i in range(N):
    diag[i,i] = b[:,i]

term = (a + w.T @ X.T) @ diag
tot2 = (term @ term.T)
print(tot, tot2)
