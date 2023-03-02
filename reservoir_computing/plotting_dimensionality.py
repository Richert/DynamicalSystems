import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from matplotlib.patches import Rectangle
import pickle
import os
import seaborn as sb
from pycobi import ODESystem
import numpy as np

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 400
plt.rcParams['figure.figsize'] = (12, 4.0)
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6


# load data
###########

# SNN simulations
path = "results/dimensionality"
fn = "rs_dimensionality"
dimensionality = []
dim_columns = ["p", "Delta", "d", "dim", "modules"]
module_examples = {}
example = 0
condition = {"Delta": 1.0, "p": 0.125, "d": 100.0}
for file in os.listdir(path):
    if fn in file:
        f = pickle.load(open(f"{path}/{file}", "rb"))
        dim = f["dim"]
        mods = f["modules"]
        p = f["sweep"]["p"]
        Delta = f["sweep"]["Delta"]
        for i in range(dim.shape[0]):
            row = []
            dim_tmp = dim.iloc[i, :]
            row.append(p)
            row.append(Delta)
            row.append(dim_tmp["d"])
            row.append(dim_tmp["dim"])
            row.append(len(mods["m"][i]))
            dimensionality.append(row)
            include_example = True
            condition_test = {"Delta": Delta, "p": p, "d": dim_tmp["d"]}
            for key in condition:
                if np.abs(condition[key] - condition_test[key]) > 1e-4:
                    include_example = False
            if include_example:
                module_examples[example] = {"m": mods["m"][i], "s": mods["s"][i], "cov": mods["cov"][i]}
                example += 1

dimensionality = pd.DataFrame(columns=dim_columns, data=dimensionality)

# bifurcation data
a = ODESystem.from_file("results/rs_bifurcations.pkl", auto_dir="~/PycharmProjects/auto-07p")

# load mean-field data
fn = "results/rs_arnold_tongue"
conditions = ["het", "hom"]
Deltas = [1.0, 0.1]
columns = ["coh", "omega", "alpha", "Delta"]
data = []
for cond, Delta in zip(conditions, Deltas):
    data_tmp = pickle.load(open(f"{fn}_{cond}.pkl", "rb"))
    alphas = np.round(data_tmp["alphas"] * 1e3, decimals=2)
    omegas = np.round(data_tmp["omegas"] * 1e3, decimals=2)
    coh = data_tmp["coherence"]
    for i, alpha in enumerate(alphas):
        for j, omega in enumerate(omegas):
            data.append([coh[i, j], omega, alpha, Delta])
mf_data = pd.DataFrame(data=data, columns=columns)

# Figure 1
##########

fig = plt.figure(1)
grid = GridSpec(ncols=3, nrows=2, figure=fig)
dv = "dim"
dv_label = "Dimensionality"
ticks = 3
square = False
cbar_kwargs = {"shrink": 1.0}

# plot mf coherence for the heterogeneous case
ax = fig.add_subplot(grid[0, 0])
mf_het = mf_data.loc[mf_data.loc[:, "Delta"] > 0.5, :]
sb.heatmap(mf_het.pivot(index="alpha", columns="omega", values="coh"), ax=ax, vmin=0.0, vmax=1.0, annot=False,
           cbar=True, xticklabels=ticks, yticklabels=ticks, square=square, cbar_kws=cbar_kwargs)
ax.set_xlabel(r'$\omega$ (Hz)')
ax.set_ylabel(r'$\alpha$ (Hz)')
ax.set_title(r"$\Delta_v = 1.0$ mV")

# plot mf coherence for the homogeneous case
ax = fig.add_subplot(grid[1, 0])
mf_hom = mf_data.loc[mf_data.loc[:, "Delta"] < 0.5, :]
sb.heatmap(mf_hom.pivot(index="alpha", columns="omega", values="coh"), ax=ax, vmin=0.0, vmax=1.0, annot=False,
           cbar=True, xticklabels=ticks, yticklabels=ticks, square=square, cbar_kws=cbar_kwargs)
ax.set_xlabel(r'$\omega$ (Hz)')
ax.set_ylabel(r'$\alpha$ (Hz)')
ax.set_title(r"$\Delta_v = 0.1$ mV")

# d = 10
ax = fig.add_subplot(grid[0, 1])
a.plot_continuation(f'PAR(8)', f'PAR(5)', cont=f'Delta/eta:lp1', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E')
a.plot_continuation(f'PAR(8)', f'PAR(5)', cont=f'Delta/eta:lp2', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E')
ax.set_xlabel(r"$\eta$ (pA)")
ax.set_ylabel(r"$\Delta_v$ (mV)")
ax.set_title(r"$\kappa = 10$ pA")
ax.set_xlim([20.0, 70.0])
ax.set_ylim([0.0, 1.6])
ax.axhline(y=0.1, color='black', alpha=0.5, linestyle='--', linewidth=0.5)
ax.axhline(y=0.5, color='black', alpha=0.5, linestyle='--', linewidth=0.5)
ax.axhline(y=1.0, color='black', alpha=0.5, linestyle='--', linewidth=0.5)
ax.axvline(x=50.0, color='black', alpha=0.5, linestyle='--', linewidth=0.5)

# d = 100
ax = fig.add_subplot(grid[1, 1])
a.plot_continuation(f'PAR(8)', f'PAR(5)', cont=f'Delta/eta:lp3', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_unstable="solid")
a.plot_continuation(f'PAR(8)', f'PAR(5)', cont=f'Delta/eta:lp4', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E')
a.plot_continuation(f'PAR(8)', f'PAR(5)', cont=f'Delta/eta:hb1', ax=ax, line_color_stable='#148F77',
                    line_color_unstable='#148F77')
ax.set_xlabel(r"$\kappa$ (pA)")
ax.set_ylabel(r"$\Delta_v$ (mV)")
ax.set_title(r"$\kappa = 100$ pA")
ax.set_xlim([20.0, 70.0])
ax.set_ylim([0.0, 1.6])
ax.axhline(y=0.1, color='black', alpha=0.5, linestyle='--', linewidth=0.5)
ax.axhline(y=0.5, color='black', alpha=0.5, linestyle='--', linewidth=0.5)
ax.axhline(y=1.0, color='black', alpha=0.5, linestyle='--', linewidth=0.5)
ax.axvline(x=55.0, color='black', alpha=0.5, linestyle='--', linewidth=0.5)

# plot dimensionality bar graph for steady-state regime
dim_ss = dimensionality.loc[dimensionality["d"] < 50.0, :]
ax = fig.add_subplot(grid[0, 2])
sb.barplot(data=dim_ss, x="p", y=dv, hue="Delta", errorbar="sd", palette="dark", alpha=0.8, ax=ax)
ax.set_ylabel(dv_label)
ax.set_title(r"Asynchronous regime ($\kappa = 10$ pA)")

# plot dimensionality bar graph for steady-state regime
dim_lc = dimensionality.loc[dimensionality["d"] > 50.0, :]
ax = fig.add_subplot(grid[1, 2])
g = sb.barplot(data=dim_lc, x="p", y=dv, hue="Delta", errorbar="sd", palette="dark", alpha=0.8, ax=ax)
ax.set_ylabel(dv_label)
ax.set_title(r"Synchronous regime ($\kappa = 100$ pA)")

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/dimensionality.pdf')

# Figure 2
##########

fig = plt.figure(2)
grid = GridSpec(ncols=2, nrows=2, figure=fig)
module_example = 1

# plot the modularity example
modex = module_examples[module_example]
ax = fig.add_subplot(grid[0, 3])
ax.imshow(modex["cov"], cmap='magma', aspect="equal", interpolation="none")
ax.set_xlabel('neuron id')
ax.set_ylabel('neuron id')
ax.set_title(fr"$p = {condition['p']}$, $\Delta_v = {condition['Delta']}$ mV, $\kappa = {condition['d']}$ pA")

ax2 = fig.add_subplot(grid[1, 3])
cmap = cm.get_cmap('tab10')
xl = (2000, 3000)
x, y = 0, 0
for key, (indices, _) in modex["m"].items():
    ax2.plot(modex["s"]["time"][xl[0]:xl[1]], modex["s"][key][xl[0]:xl[1]], c=cmap(key-1))
    inc = len(indices)
    rect = Rectangle([x, y], inc, inc, edgecolor=cmap(key - 1), facecolor='none')
    ax.add_patch(rect)
    x += inc
    y += inc
ax2.set_xlabel('time (ms)')
ax2.set_ylabel(r'$s$')
ax2.set_title("Communities exhibit distinct \n mean-field dynamics")