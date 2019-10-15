from multiprocessing import Pool
import numpy as np

np.random.seed(123)

def test(dummy):
    N = np.random.randint(5, 50)    # Number of datapoints
    p = np.random.randint(2, 6)     # Number of features
    X = np.random.random((N,p))     # Matrix x
    W = np.random.random((p,1))     # Vector w
    P = np.random.random((1,N))     # Vector p
    Q = np.random.random((1,N))     # Vector q

    # Evaluating the summation form
    summation_total = 0
    for i in range(N):
        x_i = X[i,:,np.newaxis]     # Vector x_i
        summation_total += (Q[:,i]**2)*((P[:,i] - W.T @ x_i)**2)

    root_R = np.zeros((N,N))
    for i in range(N):
        root_R[i,i] = Q[:,i]
    R = root_R @ root_R             # Matrix r

    # Evaluating the vector form
    vector_total = (P - W.T @ X.T) @ R @ (P - W.T @ X.T).T

    # Calculating and saving the difference between each total
    difference = np.squeeze(np.abs(summation_total - vector_total))
    return difference

N_tests = 1E4

# Running tests a total of "N_tests" times
pool = Pool()
results = np.array(pool.map(test, (None for i in range(int(N_tests)))))

# Gathering information on the results and printing
max_res, mean_res, median_res = \
np.max(results), np.mean(results), np.median(results)
print("Information on differences between summation total and vector total")
print(f"\tMaximum: {max_res}\n\tMean: {mean_res}\n\tMedian: {median_res}")
