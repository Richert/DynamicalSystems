import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle
import os
import numpy as np
import pandas as pd
import seaborn as sb
import sys

# file path and name
path = "results/bump"
fn = "snn_bump"

# parameters of interest
p1 = "Delta"
p2 = "p_in"

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6


# load data
sweep = pickle.load(open("config/bump_sweep.pkl", "rb"))
precisions = {"within_bump_dist": 4, "outside_bump_dist": 4, p1: 2, p2: 3}
example_condition = {p1: [0.2, 0.8, 1.6]}
for i, val in enumerate(example_condition[p1]):
    idx = np.argmin(np.abs(sweep[p1] - val)).squeeze()
    example_condition[p1][i] = np.round(sweep[p1][idx], decimals=precisions[p1])
example_trials = [0, 1]
examples_single = {val: {p2: [], "s": [], "neuron_id": [], "target_lb": [], "target_ub": []}
                   for val in example_condition[p1]}

for file in os.listdir(path):
    if fn in file:

        # extract data
        data = pickle.load(open(f"{path}/{file}", "rb"))
        targets = data["target_dists"]
        snn_data = data["population_dists"]
        bumps = data["bumps"]
        v1 = np.round(data["sweep"][p1], decimals=precisions[p1])
        trial = data["sweep"]["trial"]
        v2s = np.round(data[p2], decimals=precisions[p2])

        if trial in example_trials and v1 in example_condition[p1]:

            examples_tmp = {key: [] for key in examples_single[v1]}

            # collect bump example
            for snn, target, v2 in zip(snn_data, targets, v2s):
                target = np.diff(target)
                target_lb = np.argwhere(target > 1e-12).squeeze()
                target_ub = np.argwhere(target < 0.0).squeeze()
                try:
                    target_lb = int(target_lb)
                    target_ub = int(target_ub)
                except TypeError:
                    target_lb = 0
                    target_ub = len(target)-1
                for i, s in enumerate(snn):
                    examples_tmp["s"].append(s / np.max(snn))
                    examples_tmp["neuron_id"].append(i)
                    examples_tmp["target_lb"].append(target_lb)
                    examples_tmp["target_ub"].append(target_ub)
                    examples_tmp[p2].append(v2)

            for key, val in examples_tmp.items():
                examples_single[v1][key].append(val)

# plotting
##########

# create figure layout
fig = plt.figure(1)
grid = GridSpec(nrows=2, ncols=3, figure=fig)

# meta parameters
ticks = 5

# plot first example bumps over p_in
example_id = 0
titles = [r"(A) Example bumps for $\Delta_{rs} = 0.2$ mV",
          r"(B) Example bumps for $\Delta_{rs} = 0.8$ mV",
          r"(C) Example bumps for $\Delta_{rs} = 1.6$ mV"]
p_in_ticks = [0.02, 0.1, 0.5]
for i, val in enumerate(example_condition[p1]):
    ax = fig.add_subplot(grid[0, i])
    example = pd.DataFrame.from_dict({key: v[example_id] for key, v in examples_single[val].items()})
    data = example.pivot(index=p2, columns="neuron_id", values="s")
    sb.heatmap(data, cbar=True, ax=ax, xticklabels=100*ticks, yticklabels=ticks, rasterized=True)
    lbs = example.pivot(index=p2, columns="neuron_id", values="target_lb").iloc[:, 0]
    ubs = example.pivot(index=p2, columns="neuron_id", values="target_ub").iloc[:, 0]
    for j, (lb, ub) in enumerate(zip(lbs.values, ubs.values)):
        ax.plot([lb, lb], [j, j+1], color="blue", linewidth=1)
        ax.plot([ub, ub], [j, j+1], color="blue", linewidth=1)
    ax.set_title(titles[i])
    ax.set_xlabel("neuron id")
    ax.set_ylabel(r"$p_{in}$")
    ylocs = [np.argmin(np.abs(data.index - tick)).squeeze() for tick in p_in_ticks]
    ax.set_yticks(ylocs, labels=[str(tick) for tick in p_in_ticks])

# plot second example bumps over p_in
example_id = 1
p_in_ticks = [0.02, 0.1, 0.5]
for i, val in enumerate(example_condition[p1]):
    ax = fig.add_subplot(grid[1, i])
    example = pd.DataFrame.from_dict({key: v[example_id] for key, v in examples_single[val].items()})
    data = example.pivot(index=p2, columns="neuron_id", values="s")
    sb.heatmap(data, cbar=True, ax=ax, xticklabels=100*ticks, yticklabels=ticks, rasterized=True)
    lbs = example.pivot(index=p2, columns="neuron_id", values="target_lb").iloc[:, 0]
    ubs = example.pivot(index=p2, columns="neuron_id", values="target_ub").iloc[:, 0]
    for j, (lb, ub) in enumerate(zip(lbs.values, ubs.values)):
        ax.plot([lb, lb], [j, j+1], color="blue", linewidth=1)
        ax.plot([ub, ub], [j, j+1], color="blue", linewidth=1)
    ax.set_xlabel("neuron id")
    ax.set_ylabel(r"$p_{in}$")
    ylocs = [np.argmin(np.abs(data.index - tick)).squeeze() for tick in p_in_ticks]
    ax.set_yticks(ylocs, labels=[str(tick) for tick in p_in_ticks])

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/SI_bump.svg')
plt.show()
