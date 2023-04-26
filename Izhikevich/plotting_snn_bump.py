import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle
import os
import numpy as np
import pandas as pd
import seaborn as sb


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
plt.rcParams['figure.figsize'] = (6, 8)
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6


# load data
sweep = pickle.load(open("config/bump_sweep.pkl", "rb"))
precisions = {"within_bump_dist": 4, "outside_bump_dist": 4, p1: 2, p2: 3}
example_condition = {p1: [0.1, 1.0]}
for i, val in enumerate(example_condition[p1]):
    idx = np.argmin(np.abs(sweep[p1] - val)).squeeze()
    example_condition[p1][i] = np.round(sweep[p1][idx], decimals=precisions[p1])
single_id = 1
examples_avg = {val: {p2: [], "s": [], "neuron_id": [], "target_lb": [], "target_ub": []}
                for val in example_condition[p1]}
examples_single = examples_avg.copy()
results = {"within_bump_dist": [], "outside_bump_dist": [], p1: [], p2: []}

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

        # store KLD results
        for key in ["within_bump_dist", "outside_bump_dist", p1, p2]:
            results[key].append(np.round(bumps[key], decimals=precisions[key]))

        # store examples
        if v1 in example_condition[p1]:

            examples_tmp = {key: [] for key in examples_avg[v1]}

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

            # collect examples for averaging
            for key, val in examples_tmp.items():
                examples_avg[v1][key].append(val)

            if trial == single_id:

                # for single bump examples
                examples_single[v1] = examples_tmp.copy()

# create final results data frames
results_final = {key: [] for key in results}
for v1 in sweep[p1]:
    idx = np.argwhere([results[p1][i][0] for i in range(len(results[p1]))] == np.round(v1, decimals=precisions[p1])
                      ).squeeze()
    for key, val in results.items():
        results_final[key].extend(list(np.round(np.mean([val[i] for i in idx], axis=0), decimals=precisions[key])))
results = pd.DataFrame.from_dict(results_final)

# plotting
##########

# create figure layout
fig = plt.figure(1)
grid = GridSpec(nrows=8, ncols=2, figure=fig)

# meta parameters
example_id = 0
ticks = 5

# plot time series
conditions = ["hom", "het"]
titles = [r"(B) $p_{in} = 0.25$, $\Delta_{rs} = 0.1$ mV",
          r"(C) $p_{in} = 0.25$, $\Delta_{rs} = 1.0$ mV"]
subplots = [0, 1]
for idx, cond, title in zip(subplots, conditions, titles):
    ax = fig.add_subplot(grid[idx, 1])
    data = pickle.load(open(f"results/snn_bump_{cond}.pkl", "rb"))["results"]
    ax.imshow(data.T, interpolation="none", aspect="auto", cmap="Greys")
    ax.set_xlabel("time")
    ax.set_ylabel("neuron id")
    ax.set_title(title)

# plot example bumps over p_in
titles = [r"(D) Example bump for $\Delta_{rs} = 0.1$ mV",
          r"(E) Example bump for $\Delta_{rs} = 1.0$ mV"]
p_in_ticks = [0.02, 0.1, 0.5]
for i, val in enumerate(example_condition[p1]):
    ax = fig.add_subplot(grid[2:4, i])
    example = pd.DataFrame.from_dict({key: np.asarray(v) for key, v in examples_single[val].items()})
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

# plot average bumps over p_in
titles = [r"(F) Average bump for $\Delta_{rs} = 0.1$ mV",
          r"(G) Average bump for $\Delta_{rs} = 1.0$ mV"]
for i, val in enumerate(example_condition[p1]):
    ax = fig.add_subplot(grid[4:6, i])
    example_dict = {}
    for key, v in examples_avg[val].items():
        v_mean = np.mean(v, axis=0)
        if key in precisions:
            v_final = np.round(v_mean, decimals=precisions[key])
        elif key == "neuron_id":
            v_final = np.asarray(v_mean, dtype=np.int32)
        else:
            v_final = v_mean
        example_dict[key] = v_final
    example = pd.DataFrame.from_dict(example_dict)
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

# plot within-bump distance
ax = fig.add_subplot(grid[6:, 0])
within_dist = results.pivot(index=p2, columns=p1, values="within_bump_dist")
sb.heatmap(within_dist, cbar=True, ax=ax, xticklabels=ticks, yticklabels=ticks, rasterized=True)
ax.set_title("(H) RMSE (input, bump)")
ax.set_ylabel(r"$p_{in}$")
ax.set_xlabel(r"$\Delta_{rs}$")
ylocs = [np.argmin(np.abs(within_dist.index - tick)).squeeze() for tick in p_in_ticks]
ax.set_yticks(ylocs, labels=[str(tick) for tick in p_in_ticks])

# plot outside-bump distance
ax = fig.add_subplot(grid[6:, 1])
outside_dist = results.pivot(index=p2, columns=p1, values="outside_bump_dist")
sb.heatmap(outside_dist, cbar=True, ax=ax, xticklabels=ticks, yticklabels=ticks, rasterized=True)
ax.set_xlim([0, 19])
ax.set_title("(I) Average outside-bump activity")
ax.set_ylabel(r"$p_{in}$")
ax.set_xlabel(r"$\Delta_{rs}$")
ylocs = [np.argmin(np.abs(outside_dist.index - tick)).squeeze() for tick in p_in_ticks]
ax.set_yticks(ylocs, labels=[str(tick) for tick in p_in_ticks])

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/snn_bump.pdf')
plt.show()
