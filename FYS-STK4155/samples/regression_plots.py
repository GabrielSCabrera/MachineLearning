import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("..")
from utils.classes import Regression

N = int(1E3)
x = np.linspace(0, 1, N) + np.random.normal(0, 10, N)
y = np.linspace(0, 1, N) + np.random.normal(0, 10, N)
z = 0.1*x**3 + 0.2*x**2*y**2 - 0.1*x*y**2 - 1.2*np.sin(x) + np.random.normal(0, 1, N)

X = np.array([x,y]).T

R = Regression(X,z)
R.poly(4)
R.plot(detail = 0.5)
# R.reset()
R.split(2)
R.poly(3)
print(f"MSE = {R.mse():10.4g}")
print(f"R²  = {R.r_squared():10.4g}")
print(f"Var = \n\n{R.variance(1)}")

R2, MSE, variance = R.k_fold(k = 20, degree = 3, sigma = 1)

print("\nK-Fold Mean Values:\n\n")
print(f"MSE = {np.mean(MSE):10.4g}")
print(f"R²  = {np.mean(R2):10.4g}")
print(f"Var = \n\n{np.mean(variance, axis = 0)}")