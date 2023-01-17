import sys
import os
import pickle
import numpy as np
import pandas as pd
from matplotlib import gridspec
import matplotlib.pyplot as plt

# settings
##########

print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6
cmap = plt.get_cmap('plasma')

# prepare data
##############

path = "results/ir_delta"
fn = "ir_delta_46"
data = pickle.load(open(f"{path}/{fn}.pkl", "rb"))
print(f"Condition: {data['sweep']}")

# plotting
##########

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=3, ncols=2, figure=fig)

# timeseries
trial = 1
start, stop = 0, 3000
ax = fig.add_subplot(grid[0, :])
ax.plot(np.mean(data["s"][trial].iloc[start:stop, :].values, axis=1), color="blue")
ax2 = ax.twinx()
ax2.plot(data["I_ext"][::data["sr"], 0][start:stop], color="orange")
ax.set_xlabel("time (ms)")
ax.set_ylabel("s")
ax2.set_ylabel("I")

ax = fig.add_subplot(grid[1, :])
im = ax.imshow(np.sqrt(data["s"][trial].iloc[start:stop, :].values.T), aspect=0.4)
ax.set_xlabel("time (ms)")
ax.set_ylabel("s")
plt.colorbar(im, ax=ax)

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.show()
