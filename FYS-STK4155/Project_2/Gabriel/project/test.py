import subprocess, os
import numpy as np

lrs = np.linspace(0.005,0.2, 15)
try:
    os.mkdir("series")
except FileExistsError:
    pass

for lr in lrs:
    subprocess.call(['python3', 'credit_card.py', 'save=results_series3_',
                     f'lr={lr}', f'saveimg=series/{lr}lr.pdf',
                     'epochs=256', "reg=0.0001"])
