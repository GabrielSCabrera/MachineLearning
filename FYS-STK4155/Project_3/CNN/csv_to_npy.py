"""
    Imports the EMNIST .csv data in 'data_dir' and converts each split into a
    .npy file of unsigned 8-bit integers, and saves them to 'data_dir'.
"""

from multiprocessing import Pool
from time import time
import numpy as np
import config
import re

# READING DATA
def read_csv(filename):
    with open(filename, "r") as infile:
        lines = infile.readlines()
    N = len(lines)
    p = len(lines[0].strip().split(","))
    lines = "".join(lines)
    lines = re.sub(r"\n", r",", lines).split(",")[:-1]
    lines = np.array(lines, dtype = np.uint8).reshape(N,p)
    return lines

if __name__ == "__main__":
    pool = Pool()
    t0 = time()
    datasets = pool.map(read_csv, config.csv_names)
    for dataset, name in zip(datasets, config.npy_names):
        np.save(name, dataset)
    print(time()-t0)
