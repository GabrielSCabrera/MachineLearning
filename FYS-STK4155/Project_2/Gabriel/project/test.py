from multiprocessing import Pool
import subprocess, os
import numpy as np

regs = np.logspace(-8, -2, 10)
lrs = np.linspace(0.005, 0.2, 10)
epochs = 200
gpu = True

def execute(lr, reg, dirname, epochs, gpu, filename, lrcount, regcount, savename):
    loc = f"{dirname}/lr_{lrcount}_reg_{regcount}"
    try:
        os.mkdir(loc)
    except FileExistsError:
        pass
    subprocess.call(['python3', filename,
                    f'save={loc}', f'lr={lr}', f'saveimg={savename}',
                    f'epochs={epochs}', f"reg={reg}", f"GPU={gpu}"])

try:
    os.mkdir("grid_cc")
except FileExistsError:
    pass

try:
    os.mkdir("grid_franke")
except FileExistsError:
    pass

for m,lr in enumerate(lrs):
    for n,reg in enumerate(regs):
        execute(lr, reg, "grid_franke", epochs, gpu, 'franke.py',m,n, "franke.png")
        execute(lr, reg, "grid_cc", epochs, gpu, 'credit_card.py',m,n, "roc.png")
