import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1)

dtype = np.float64
m = 30
start = -2
stop = 4

eps = 1
r = np.random.rand(m)*eps
deg = 16

x = np.linspace(start, stop, m, dtype = dtype)
y1 = x*(np.cos(r + 0.5*x**3)+np.sin(0.5*x**3))
y2 = 4*x**5 - 5*x**4 - 20*x**3 + 10*x**2 + 40*x + 10 + r

def backward_substitution(u, y):
    n = u.shape[0]
    x = np.zeros(n)
    x[-1] = y[-1]/u[-1,-1]
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(u[i,i+1:], x[i+1:]))/u[i,i]
    return x

def forward_substitution(L, b):
    n = L.shape[0]
    y = np.zeros(n)
    y[0] = b[0]/L[0,0]
    for i in range(1, n):
        y[i] = (b[i] - np.dot(L[i,:i], y[:i]))/L[i,i]
    return y

def design(x, deg):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    m = x.shape[0]
    exponents = np.tile(np.arange(0, deg+1, 1), (m,1))
    A = np.array([x]).T
    A = np.tile(A, deg+1)
    A = A**exponents
    return A

def cholesky_decomposition(x):
    A = np.array(x, dtype = dtype)
    n = A.shape[0]
    L = np.zeros_like(A, dtype = dtype)
    D = np.zeros_like(A, dtype = dtype)
    for k in range(n):
        L[:,k] = A[:,k]/A[k,k]
        D[k,k] = A[k,k]
        A = A - (D[k,k] * np.outer(L[:,k], L[:,k]))
    R = L @ np.sqrt(D)
    return R

def betas_QR(x, y, deg):
    A = design(x, deg)

    # QR Factorization
    Q,R = np.linalg.qr(A)
    C = Q.T @ y

    # Solving System of Equations
    beta = backward_substitution(R, C)
    solve(beta, x)
    return beta

def betas_cholesky(x, b, deg):
    A = design(x, deg)

    # Cholesky Decomposition
    B = A.T @ A
    R = cholesky_decomposition(B)

    # Solving System of Equations
    y = forward_substitution(R, A.T @ b)
    beta = backward_substitution(R.T, y)
    return beta

def solve(beta, x):
    deg = len(beta)-1
    A = design(x, deg)
    y = A @ beta
    return y

def mse(beta, x, y):
    return np.mean((y - solve(beta, x))**2)

def plot_MSE(x, y, max_deg):
    start = x[0]
    stop = x[-1]

    test_size = 15

    N_test = int(round(len(x)*test_size*0.01))
    test_idx = np.random.choice(a = N_test, size = N_test, replace = False)
    x_test = x[test_idx]
    y_test = y[test_idx]
    x_train = np.delete(x, test_idx, axis = 0)
    y_train = np.delete(y, test_idx)

    MSE_QR = np.zeros(max_deg)
    MSE_CH = np.zeros(max_deg)
    deg = np.arange(1, max_deg + 1, 1)

    for d in range(max_deg):
        beta_QR = betas_QR(x_train, y_train, d + 1)
        beta_CH = betas_cholesky(x_train, y_train, d + 1)

        MSE_QR[d] = mse(beta_QR, x_test, y_test)
        MSE_CH[d] = mse(beta_CH, x_test, y_test)

    plt.semilogy(deg, np.abs(MSE_QR-MSE_CH))
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Absolute Difference in MSE for QR and Cholesky Decomposition")
    plt.xlim([1, max_deg])
    # plt.show()
    plt.savefig("MSE.pdf")
    plt.close()


def plot_QR(x, y, deg):

    start = x[0]
    stop = x[-1]

    # Plotting
    N_plot = 5E2

    x_plot = np.linspace(start, stop, int(N_plot))

    test_size = 33

    N_test = int(round(len(x)*test_size*0.01))
    test_idx = np.random.choice(a = N_test, size = N_test, replace = False)
    x_test = x[test_idx]
    y_test = y[test_idx]
    x_train = np.delete(x, test_idx, axis = 0)
    y_train = np.delete(y, test_idx)

    beta = betas_QR(x_train, y_train, deg)
    y_plot = solve(beta, x_plot)

    plt.plot(x, y, "o")
    plt.plot(x_plot, y_plot)

    digit = int(str(deg)[-1])
    if deg == 1:
        end = "st"
    elif digit == 2:
        end = "nd"
    elif digit == 3:
        end = "rd"
    else:
        end = "th"

    plt.legend(["Input Data", f"{deg}{end} Degree\nPolynomial Regression"])
    plt.xlim([start, stop])
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.show()
    # plt.savefig(f"QR_{deg:d}.pdf")
    # plt.close()

def plot_cholesky(x, y, deg):

    start = x[0]
    stop = x[-1]

    # Plotting
    N_plot = 5E2

    x_plot = np.linspace(start, stop, int(N_plot))

    test_size = 33

    N_test = int(round(len(x)*test_size*0.01))
    test_idx = np.random.choice(a = N_test, size = N_test, replace = False)
    x_test = x[test_idx]
    y_test = y[test_idx]
    x_train = np.delete(x, test_idx, axis = 0)
    y_train = np.delete(y, test_idx)

    beta = betas_cholesky(x_train, y_train, deg)
    y_plot = solve(beta, x_plot)

    plt.plot(x, y, "o")
    plt.plot(x_plot, y_plot)

    digit = int(str(deg)[-1])
    if deg == 1:
        end = "st"
    elif digit == 2:
        end = "nd"
    elif digit == 3:
        end = "rd"
    else:
        end = "th"

    plt.legend(["Input Data", f"{deg}{end} Degree\nPolynomial Regression"])
    plt.xlim([start, stop])
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.show()
    # plt.savefig(f"CH_{deg:d}.pdf")
    # plt.close()

if __name__ == "__main__":
    # plot_QR(x, y1, deg)
    # plot_cholesky(x, y1, deg)
    plot_MSE(x, y1, deg)
