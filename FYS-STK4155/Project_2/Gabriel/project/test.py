import subprocess, os
import numpy as np

regparams = np.logspace(-16,-2, 15)
try:
    os.mkdir("series")
except FileExistsError:
    pass

for reg in regparams:
    subprocess.call(['python3', 'credit_card.py', 'save=results_series2_',
                     f'reg={reg}', f'saveimg=series/{reg}reg.pdf',
                     'epochs=256'])
