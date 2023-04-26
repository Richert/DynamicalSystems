import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle
import os
import numpy as np
import pandas as pd
import seaborn as sb
from pycobi import ODESystem
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
plt.rcParams['figure.figsize'] = (6, 8)
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6


# load data
sweep = pickle.load(open("config/bump_sweep.pkl", "rb"))
precisions = {"within_bump_dist": 4, "outside_bump_dist": 4, p1: 2, p2: 3}
example_condition = {p1: [0.2, 1.5]}
for i, val in enumerate(example_condition[p1]):
    idx = np.argmin(np.abs(sweep[p1] - val)).squeeze()
    example_condition[p1][i] = np.round(sweep[p1][idx], decimals=precisions[p1])
single_id = 0
examples_avg = {val: {p2: [], "s": [], "neuron_id": [], "target_lb": [], "target_ub": []}
                for val in example_condition[p1]}
examples_single = examples_avg.copy()
results = {"within_bump_dist": [], "outside_bump_dist": [], p1: [], p2: []}

# load pyauto data
auto_dir = "~/PycharmProjects/auto-07p"
a_rs = ODESystem.from_file(f"results/rs.pkl", auto_dir=auto_dir)

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

# plot bifurcation diagram
ax = fig.add_subplot(grid[:2, 0])
a_rs.plot_continuation('PAR(8)', 'PAR(5)', cont=f'D/I:lp1', ax=ax, line_color_stable='#5D6D7E',
                       line_color_unstable='#5D6D7E', line_style_unstable='solid')
a_rs.plot_continuation('PAR(8)', 'PAR(5)', cont=f'D/I:lp2', ax=ax, line_color_stable='#5D6D7E',
                       line_color_unstable='#5D6D7E', line_style_unstable='solid')
ax.set_ylabel(r'$\Delta_{rs}$ (mv)')
ax.set_xlabel(r'$I_{rs}$ (pA)')
ax.set_title(r'(A) RS: $\kappa_{rs} = 10.0$ pA')
ax.set_ylim([0.0, 4.0])
ax.set_xlim([10.0, 70.0])
ax.axvline(x=30.0, color='black', alpha=0.5, linestyle='--', linewidth=0.5)
ax.axvline(x=60.0, color='red', alpha=0.5, linestyle='--', linewidth=0.5)

# plot time series
conditions = ["hom", "het"]
titles = [r"(B) $p_{in} = 0.25$, $\Delta_{rs} = 0.2$ mV",
          r"(C) $p_{in} = 0.25$, $\Delta_{rs} = 1.5$ mV"]
subplots = [0, 1]
for idx, cond, title in zip(subplots, conditions, titles):
    ax = fig.add_subplot(grid[idx, 1])
    data = pickle.load(open(f"results/snn_bump_{cond}.pkl", "rb"))["results"]
    ax.imshow(data.T, interpolation="none", aspect="auto", cmap="Greys")
    if idx == 1:
        ax.set_xlabel("time (ms)")
        # ax.set_ylabel("neuron id")
    ax.set_xticks([0, 10000, 20000], labels=["0", "1000", "2000"])
    ax.set_title(title)

# plot example bumps over p_in
titles = [r"(D) Example bump for $\Delta_{rs} = 0.2$ mV",
          r"(E) Example bump for $\Delta_{rs} = 1.5$ mV"]
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
titles = [r"(F) Average bump for $\Delta_{rs} = 0.2$ mV",
          r"(G) Average bump for $\Delta_{rs} = 1.5$ mV"]
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
y_pos = np.argmin(np.abs(within_dist.index - 0.25)).squeeze() + 0.5
x_pos1 = np.argmin(np.abs(within_dist.columns.values - example_condition[p1][0])).squeeze() + 0.5
x_pos2 = np.argmin(np.abs(within_dist.columns.values - example_condition[p1][1])).squeeze() + 0.5
ax.axvline(x=x_pos1, color='white', linestyle='--', linewidth=1)
ax.axvline(x=x_pos2, color='blue', linestyle='--', linewidth=1)
ax.plot([x_pos1-0.25, x_pos1+0.25], [y_pos, y_pos], color="white", linewidth=1)
ax.plot([x_pos2-0.25, x_pos2+0.25], [y_pos, y_pos], color="blue", linewidth=1)

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
y_pos = np.argmin(np.abs(outside_dist.index - 0.25)).squeeze() + 0.5
x_pos1 = np.argmin(np.abs(outside_dist.columns.values - example_condition[p1][0])).squeeze() + 0.5
x_pos2 = np.argmin(np.abs(outside_dist.columns.values - example_condition[p1][1])).squeeze() + 0.5
ax.axvline(x=x_pos1, color='white', linestyle='--', linewidth=1)
ax.axvline(x=x_pos2, color='blue', linestyle='--', linewidth=1)
ax.plot([x_pos1-0.25, x_pos1+0.25], [y_pos, y_pos], color="white", linewidth=1)
ax.plot([x_pos2-0.25, x_pos2+0.25], [y_pos, y_pos], color="blue", linewidth=1)

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/snn_bump.svg')
plt.show()
