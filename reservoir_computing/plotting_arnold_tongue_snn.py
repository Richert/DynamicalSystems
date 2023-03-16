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
plt.rcParams['figure.dpi'] = 400
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6

# load snn data
path = "/media/richard/data/entrainment_snn"
fn = "rs_entrainment"
snn_data = []
columns = ["Delta", "omega", "dim", "coh_inp", "coh_noinp"]
covariance_examples = []
example = 0
condition = {"p_in": 0.1, "alpha": 0.01}
for file in os.listdir(path):
    if fn in file:
        f = pickle.load(open(f"{path}/{file}", "rb"))
        entrainment = f["entrainment"]
        cov = f["cov"]
        p_in = f["sweep"]["p_in"]
        alpha = f["sweep"]["alpha"]
        for i in range(entrainment.shape[0]):
            try:
                row = []
                entrain = entrainment.iloc[i, :]
                row.append(p_in)
                row.append(alpha)
                for c in columns:
                    row.append(entrain[c])
                snn_data.append(row)
                include_example = True
                condition_test = {"p_in": p_in, "alpha": alpha}
                for key in condition:
                    if np.abs(condition[key] - condition_test[key]) > 1e-4:
                        include_example = False
                if include_example:
                    covariance_examples.append({"cov": cov["cov"][i], "Delta": cov["Delta"][i], "omega": cov["omega"][i]})
                    example += 1
            except KeyError:
                pass

snn_data = pd.DataFrame(columns=["p_in", "alpha"] + columns, data=snn_data)

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

############
# plotting #
############

fig = plt.figure(1, figsize=(12, 6))
grid = GridSpec(ncols=4, nrows=2, figure=fig)
square = False
ticks = 3
cbar_kwargs = {"shrink": 1.0}
cols = ["(A) Mean-field model", "(B) Driven Neurons", "(C) Non-Driven Neurons", "(D) Network Dimensionality"]
rows = [rf"$\Delta = {d}$" for d in Deltas]
pad = 5.0
axes = []
for row_idx, row_label in enumerate(rows):
    row = []
    for col_idx, col_label in enumerate(cols):
        ax = fig.add_subplot(grid[row_idx, col_idx])
        if row_idx == 0:
            ax.annotate(col_label, xy=(0.5, 1), xytext=(0, pad),
                        xycoords='axes fraction', textcoords='offset points',
                        size='large', ha='center', va='baseline')
        if col_idx == 0:
            ax.annotate(row_label, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad-pad, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center')
        row.append(ax)
    axes.append(row)

# plot mean-field results
#########################

# plot coherence for the heterogeneous case
ax = axes[0][0]
mf_het = mf_data.loc[mf_data.loc[:, "Delta"] > 0.5, :]
sb.heatmap(mf_het.pivot(index="alpha", columns="omega", values="coh"), ax=ax, vmin=0.0, vmax=1.0, annot=False,
           cbar=True, xticklabels=ticks, yticklabels=ticks, square=square, cbar_kws=cbar_kwargs)
ax.set_xlabel(r'$\omega$ (Hz)')
ax.set_ylabel(r'$\alpha$ (Hz)')

# plot coherence for the homogeneous case
ax = axes[1][0]
mf_hom = mf_data.loc[mf_data.loc[:, "Delta"] < 0.5, :]
sb.heatmap(mf_hom.pivot(index="alpha", columns="omega", values="coh"), ax=ax, vmin=0.0, vmax=1.0, annot=False,
           cbar=True, xticklabels=ticks, yticklabels=ticks, square=square, cbar_kws=cbar_kwargs)
ax.set_xlabel(r'$\omega$ (Hz)')
ax.set_ylabel(r'$\alpha$ (Hz)')

# plot SNN results
##################

# plot average coherence between driven neurons and driving signal
for idx, Delta in enumerate(Deltas):
    ax = axes[idx][1]
    snn = snn_data.loc[np.abs(snn_data.loc[:, "Delta"] - Delta) < 1e-3, :]
    sb.heatmap(snn.pivot_table(index="alpha", columns="p_in", values="coh_inp"), ax=ax, vmin=0.0, vmax=1.0, annot=False,
               cbar=True, xticklabels=ticks, yticklabels=ticks, square=square, cbar_kws=cbar_kwargs)
    ax.set_xlabel(r'$p_{in}$')
    ax.set_ylabel(r'$\alpha$ (Hz)')

# plot average coherence between undriven neurons and driving signal for the 2D parameter sweep
for idx, Delta in enumerate(Deltas):
    ax = axes[idx][2]
    snn = snn_data.loc[np.abs(snn_data.loc[:, "Delta"] - Delta) < 1e-3, :]
    sb.heatmap(snn.pivot_table(index="alpha", columns="p_in", values="coh_noinp"), ax=ax, vmin=0.0, vmax=1.0, annot=False,
               cbar=True, xticklabels=ticks, yticklabels=ticks, square=square, cbar_kws=cbar_kwargs)
    ax.set_xlabel(r'$p_{in}$')
    ax.set_ylabel(r'$\alpha$ (Hz)')

# plot dimensionality of the network dynamics
for idx, Delta in enumerate(Deltas):
    ax = axes[idx][3]
    snn = snn_data.loc[np.abs(snn_data.loc[:, "Delta"] - Delta) < 1e-3, :]
    sb.heatmap(snn.pivot_table(index="alpha", columns="p_in", values="dim"), ax=ax, annot=False,
               cbar=True, xticklabels=ticks, yticklabels=ticks, square=square, cbar_kws=cbar_kwargs)
    ax.set_xlabel(r'$p_{in}$')
    ax.set_ylabel(r'$\alpha$ (Hz)')

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/entrainment.pdf')
plt.show()
