import subprocess, os

epochs = [1,2,4,8,16,32,64,128,256,512,1024]
try:
    os.mkdir("series")
except FileExistsError:
    pass
for e in epochs:
    subprocess.call(['python3', 'credit_card.py', 'save=results_series_',
                     f'epochs={e}', f'saveimg=series/{e}epochs.pdf',
                     'GPU=True'])
