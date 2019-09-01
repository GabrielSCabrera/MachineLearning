import numpy as np

import sys
sys.path.append("..")
from utils.classes import Regression

N = int(1E3)
x = np.linspace(0, 10, N) + np.random.normal(0, 10, N)
y = np.linspace(0, 10, N) + np.random.normal(0, 10, N)
z = 2 + 1.04*x**2 + 0.1*x**3 -3.1*y**2 + 3*y - 2*x*y + 2*x**2*y + 100*np.cos(x)
z = z + 2*x**3 + 0.35*x*y**2
z = z + np.random.normal(0, 1, N)

X = np.array([x,y]).T

R = Regression(X,z)
R.poly(3, alpha = 0.01)
R.plot(detail = 0.5)
