from rpy2.robjects.packages import importr, data
import numpy as np

spep = importr("mlbench")
dataset = data(spep).fetch('PimaIndiansDiabetes2')

array = []
for k,v in dataset.items():
    print(v)
    array.append(np.array(v).astype(np.float64))

array = np.array(array).T.squeeze()
trimmed = []
for row in array:
    if np.all(np.isfinite(row)):
        trimmed.append(row)
trimmed = np.array(trimmed)

np.save("PimaIndiansDiabetes2", trimmed)
