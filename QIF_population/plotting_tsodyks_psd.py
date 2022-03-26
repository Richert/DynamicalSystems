from scipy.io import loadmat
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# preparations
##############
# load matlab variable
ns = [100, 300, 900]
data = [{'micro': loadmat(f'data/tsodyks_micro_psd_{i}.mat'), 'macro': loadmat(f'data/tsodyks_macro_psd_{i}.mat')}
        for i in ns]

# plot settings
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 400
plt.rcParams['figure.figsize'] = (7, 8)
plt.rcParams['font.size'] = 8.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 0.7
plt.rcParams["font.family"] = "Times New Roman"

# plotting
fig, ax = plt.subplots()
colors = ['black', 'blue', 'red']
lines = []
for i in range(len(ns)):
    micro, macro = data[i]['micro'], data[i]['macro']
    idx = micro['freq'] < 200
    ax.plot(micro['freq'][idx], micro['pow'][idx], c=colors[i])
    ax.plot(macro['freq'][idx], macro['pow'][idx], c=colors[i], linestyle='--')
plt.legend([f'N = {n}' for n in ns])
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')
plt.show()
