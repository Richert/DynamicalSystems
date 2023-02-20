import numpy as np
import pickle
import sys, os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sb

# preparations
##############

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6

# load data
path = sys.argv[-2]
file_id = sys.argv[-1]
data = []
columns = ["coh_driven", "coh_auto", "dim", "p_in", "alpha"]
meta_data = {"Delta": 0.0, "p": 1.0, "omega": 1.0}
for id, file in enumerate(os.listdir(path)):
    if file_id in file:
        f = pickle.load(open(f"{path}/{file}", "rb"))
        if id == 0:
            for key in meta_data:
                meta_data[key] = f[key]
        row = list()
        driven = 1.0 * f["entrainment"].loc[:, "input_neuron"].values
        print(np.sum(driven)/driven.shape[0])
        row.append(np.mean(f["entrainment"].loc[driven > 0, "coh"].values))
        row.append(np.mean(f["entrainment"].loc[driven < 1, "coh"].values))
        row.append(np.mean(f["dim"]))
        row.append(f["sweep"]["p_in"])
        row.append(f["sweep"]["alpha"])
        data.append(row)

data = pd.DataFrame(data=data, columns=columns)

# plotting
##########

fig = plt.figure(1, figsize=(12, 4))
grid = GridSpec(ncols=2, nrows=1, figure=fig)

# plot average coherence between driven neurons and driving signal for the 2D parameter sweep
ax = fig.add_subplot(grid[0, 0])
sb.heatmap(data.pivot("alpha", "p_in", "coh_driven"), ax=ax)

# plot average coherence between undriven neurons and driving signal for the 2D parameter sweep
ax = fig.add_subplot(grid[0, 1])
sb.heatmap(data.pivot("alpha", "p_in", "coh_auto"), ax=ax)

# plot dimensionality of the network dynamics for the 2D parameter sweep
ax = fig.add_subplot(grid[0, 2])
sb.heatmap(data.pivot("alpha", "p_in", "dim"))

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/snn_arnold_tongue.pdf')
