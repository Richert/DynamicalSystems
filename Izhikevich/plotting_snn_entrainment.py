import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle
import os
import numpy as np
import pandas as pd
import seaborn as sb


# file path and name
path = "results/entrainment"
fn = "snn_entrainment"

# parameters of interest
p1 = "Delta"
p2 = "conn_pows"

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (6, 6)
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6


# load data
sweep = pickle.load(open("config/entrainment_sweep.pkl", "rb"))
example_condition = {p1: [0.1, 1.0]}
single_id = 3
# examples_avg = {p2: [], p1: [], "s": [], "neuron_id": [], "target_lb": [], "target_ub": []}
examples_single = {p2: [], p1: [], "ss": [], "lc": [], "neuron_id": [], "target_lb": [], "target_ub": []}
results = {"dim": [], "seq": [], "corr_driven": [], "corr_nondriven": [],  "network_oscillations": [], p1: [], p2: []}
precisions = {"dim": 0, "seq": 2, p1: 2, p2: 2}
for i, val in enumerate(example_condition[p1]):
    idx = np.argmin(np.abs(sweep[p1] - val)).squeeze()
    example_condition[p1][i] = np.round(sweep[p1][idx], decimals=precisions[p1])
for file in os.listdir(path):
    if fn in file:

        # extract data
        data = pickle.load(open(f"{path}/{file}", "rb"))
        inp_indices = data["input_indices"]
        snn_ss = data["steady_state_spiking"]
        snn_lc = data["oscillatory_spiking"]
        v1 = np.round(data["sweep"][p1], decimals=precisions[p1])
        trial = data["sweep"]["trial"]
        v2s = np.round(data[p2], decimals=precisions[p2])

        # store KLD results
        for key in results:
            try:
                results[key].append(np.round(data[key], decimals=precisions[key]))
            except KeyError:
                results[key].append(np.round(data["sweep"][key], decimals=precisions[key]))

        # store examples
        if v1 in example_condition[p1] and trial == single_id:

            # # for average bump examples
            # target = np.diff(np.mean(targets, axis=0))
            # snn = np.mean(snn_data, axis=0)
            # target_lb = np.argwhere(target > 1e-4).squeeze()
            # target_ub = np.argwhere(target < 0.0).squeeze()
            # examples_avg[p2].append(v2s)
            # for i, s in enumerate(snn):
            #     examples_avg["s"].append(s / np.max(snn))
            #     examples_avg["neuron_id"].append(i)
            #     examples_avg["target_lb"].append(target_lb)
            #     examples_avg["target_ub"].append(target_ub)
            #     examples_avg[p1].append(v1)

            # for single bump examples
            for ss, lc, v2 in zip(snn_ss, snn_lc, v2s):
                target_lb = inp_indices[0]
                target_ub = inp_indices[1]
                for i in range(ss.shape[0]):
                    examples_single["ss"].append(ss[i] / np.max(ss[i]))
                    examples_single["lc"].append(lc[i])
                    examples_single["neuron_id"].append(i)
                    examples_single["target_lb"].append(target_lb)
                    examples_single["target_ub"].append(target_ub)
                    examples_single[p1].append(v1)
                    examples_single[p2].append(v2)

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
grid = GridSpec(nrows=6, ncols=2, figure=fig)

# meta parameters
example_id = 0
ticks = 3

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

# plot average bumps over p_in
# titles = [r"(D) Average Network bump for $\Delta_{rs} = 0.1$ mV",
#           r"(E) Average Network bump for $\Delta_{rs} = 1.0$ mV"]
# for i, val in enumerate(example_condition[p1]):
#     ax = fig.add_subplot(grid[2:4, i])
#     idx = np.argwhere(np.asarray(examples_avg[p1]) == val).squeeze()
#     example = pd.DataFrame.from_dict({key: np.asarray(val)[idx] for key, val in examples_avg.items()})
#     sb.heatmap(example.pivot(index=p2, columns="neuron_id", values="s"), cbar=True, ax=ax,
#                xticklabels=50*ticks, yticklabels=ticks, rasterized=True)
#     lbs = example.pivot(index=p2, columns="neuron_id", values="target_lb").iloc[:, 0]
#     ubs = example.pivot(index=p2, columns="neuron_id", values="target_ub").iloc[:, 0]
#     for j, (lb, ub) in enumerate(zip(lbs.values, ubs.values)):
#         ax.plot([lb, lb], [j, j+1], color="blue", linewidth=1)
#         ax.plot([ub, ub], [j, j+1], color="blue", linewidth=1)
#     ax.set_title(titles[i])
#     ax.set_xlabel("neuron id")
#     ax.set_ylabel(r"$p_{in}$")

# plot example bumps over p_in
titles = [r"(D) Example Network bump for $\Delta_{rs} = 0.1$ mV",
          r"(E) Example Network bump for $\Delta_{rs} = 1.0$ mV"]
for i, val in enumerate(example_condition[p1]):
    ax = fig.add_subplot(grid[2:4, i])
    idx = np.argwhere(np.asarray(examples_single[p1]) == val).squeeze()
    example = pd.DataFrame.from_dict({key: np.asarray(val)[idx] for key, val in examples_single.items()})
    sb.heatmap(example.pivot(index=p2, columns="neuron_id", values="s"), cbar=True, ax=ax,
               xticklabels=50*ticks, yticklabels=ticks, rasterized=True)
    lbs = example.pivot(index=p2, columns="neuron_id", values="target_lb").iloc[:, 0]
    ubs = example.pivot(index=p2, columns="neuron_id", values="target_ub").iloc[:, 0]
    for j, (lb, ub) in enumerate(zip(lbs.values, ubs.values)):
        ax.plot([lb, lb], [j, j+1], color="blue", linewidth=1)
        ax.plot([ub, ub], [j, j+1], color="blue", linewidth=1)
    ax.set_title(titles[i])
    ax.set_xlabel("neuron id")
    ax.set_ylabel(r"$p_{in}$")

# plot within-bump distance
ax = fig.add_subplot(grid[4:, 0])
within_dist = results.pivot(index=p1, columns=p2, values="within_bump_dist")
sb.heatmap(within_dist, cbar=True, ax=ax, xticklabels=ticks, yticklabels=ticks, rasterized=True)
ax.set_title("(F) RMSE (input, bump)")
ax.set_xlabel(r"$p_{in}$")
ax.set_ylabel(r"$\Delta_{rs}$")

# plot outside-bump distance
ax = fig.add_subplot(grid[4:, 1])
outside_dist = results.pivot(index=p1, columns=p2, values="outside_bump_dist")
sb.heatmap(outside_dist, cbar=True, ax=ax, xticklabels=ticks, yticklabels=ticks, rasterized=True)
ax.set_xlim([0, 19])
ax.set_title("(G) Average outside-bump activity")
ax.set_xlabel(r"$p_{in}$")
ax.set_ylabel(r"$\Delta_{rs}$")

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/snn_bump.pdf')
plt.show()
