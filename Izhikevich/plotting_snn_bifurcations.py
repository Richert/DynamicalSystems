import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
from pycobi import ODESystem
sys.path.append('../')
from os import walk


#############
# load data #
#############

# mean-field bifurcation analysis
rs = ODESystem.from_file(f"results/rs.pkl", auto_dir="~/PycharmProjects/auto-07p")
fs = ODESystem.from_file(f"results/fs.pkl", auto_dir="~/PycharmProjects/auto-07p")
lts = ODESystem.from_file(f"results/lts.pkl", auto_dir="~/PycharmProjects/auto-07p")

# snn bifurcations
snn_results = {key1: {key2: {"folds": [], "hopfs": [], "delta": []} for key2 in ["gauss", "lorentz"]}
               for key1 in ["rs", "fs", "lts"]}
_, _, fnames = next(walk("results/snn_bifurcations"), (None, None, []))
for f in fnames:
    data = pickle.load(open(f"results/snn_bifurcations/{f}", "rb"))
    neuron_type = f.split("_")[1]
    deltas = {"lorentz": data["Delta"], "gauss": data["SD"]}
    for distribution_type in ["lorentz", "gauss"]:
        try:
            folds = data[f"{distribution_type}_fold"]["I_ext"]
            snn_results[neuron_type][distribution_type]["folds"].append(folds)
            snn_results[neuron_type][distribution_type]["delta"].append(deltas[distribution_type])
        except KeyError:
            pass
        try:
            hopfs = data[f"{distribution_type}_hopf"]["I_ext"]
            snn_results[neuron_type][distribution_type]["hopfs"].append(hopfs)
            snn_results[neuron_type][distribution_type]["delta"].append(deltas[distribution_type])
        except KeyError:
            pass

############
# plotting #
############

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

# create figure
fig = plt.figure(figsize=(12, 6))
grid = fig.add_gridspec(ncols=2, nrows=2)

# plot rs bifurcation diagram for kappa = 10
ax = fig.add_subplot(grid[0, 0])
rs.plot_continuation("PAR(8)", "PAR(5)", cont="D/I:lp1", ignore=["UZ"], line_style_unstable="solid", ax=ax)
rs.plot_continuation("PAR(8)", "PAR(5)", cont="D/I:lp2", ignore=["UZ"], line_style_unstable="solid", ax=ax)
ax2 = ax.twinx()
for dist, color, axis in zip(["lorentz", "gauss"], ["grey", "red"], [ax, ax2]):
    fold_left = [fold[0] for fold in snn_results["rs"][dist]["folds"]]
    fold_right = [fold[1] for fold in snn_results["rs"][dist]["folds"]]
    deltas = snn_results["rs"][dist]["delta"]
    axis.scatter(fold_left, deltas, marker="+", color=color, label=dist)
    axis.scatter(fold_right, deltas, marker="+", color=color)
ax.set_xlabel(r"$I_{rs}$ (pA)")
ax.set_ylabel(r"$\Delta_{rs}$ (mV)")
ax2.set_ylabel(r"$\sigma_{rs}$ (mV)")
ax.set_title(r"Regular Spiking Neurons: $\kappa_{rs} = 10$ pA")
# ax.set_xlim([0, 80])
# ax.set_ylim([0, 2.0])
ax.legend()

# plot rs bifurcation diagram for kappa = 100
ax = fig.add_subplot(grid[0, 1])
rs.plot_continuation("PAR(8)", "PAR(5)", cont="D/I:lp3", ignore=["UZ"], line_style_unstable="solid", ax=ax)
rs.plot_continuation("PAR(8)", "PAR(5)", cont="D/I:lp4", ignore=["UZ"], line_style_unstable="solid", ax=ax)
rs.plot_continuation("PAR(8)", "PAR(5)", cont="D/I:hb1", ignore=["UZ"], line_style_unstable="solid", ax=ax)
ax.set_xlabel(r"$I_{rs}$ (pA)")
ax.set_ylabel(r"$\Delta_{rs}$ (mV)")
ax.set_title(r"Regular Spiking Neurons: $\kappa_{rs} = 100$ pA")
ax.set_xlim([30, 80])
ax.set_ylim([0, 2.0])

# plot fs bifurcation diagram
ax = fig.add_subplot(grid[1, 0])
fs.plot_continuation("PAR(16)", "PAR(6)", cont="D/I:hb1", ignore=["UZ"], line_style_unstable="solid", ax=ax)
ax2 = ax.twinx()
for dist, color, axis in zip(["lorentz", "gauss"], ["grey", "red"], [ax, ax2]):
    fold_left = [fold[0] for fold in snn_results["fs"][dist]["hopfs"]]
    fold_right = [fold[1] for fold in snn_results["fs"][dist]["hopfs"]]
    deltas = snn_results["fs"][dist]["delta"]
    axis.scatter(fold_left, deltas, marker="+", color=color, label=dist)
    axis.scatter(fold_right, deltas, marker="+", color=color)
ax.set_xlabel(r"$I_{fs}$ (pA)")
ax.set_ylabel(r"$\Delta_{fs}$ (mV)")
ax2.set_ylabel(r"$\sigma_{fs}$ (mV)")
ax.set_title(r"Fast Spiking Neurons")
# ax.set_ylim([0, 1.0])
ax.set_xlim([0, 140])
ax.legend()

# plot lts bifurcation diagram
ax = fig.add_subplot(grid[1, 1])
lts.plot_continuation("PAR(16)", "PAR(6)", cont="D/I:hb1", ignore=["UZ"], line_style_unstable="solid", ax=ax)
for dist, color in zip(["lorentz", "gauss"], ["grey", "red"]):
    hopf_left = [fold[0] for fold in snn_results["lts"][dist]["hopfs"]]
    hopf_right = [fold[1] for fold in snn_results["lts"][dist]["hopfs"]]
    deltas = snn_results["lts"][dist]["delta"]
    ax.scatter(hopf_left, deltas, marker="+", color=color, label=dist)
    ax.scatter(hopf_right, deltas, marker="+", color=color)
ax.set_xlabel(r"$I_{lts}$ (pA)")
ax.set_ylabel(r"$\Delta_{lts}$ (mV)")
ax.set_title(r"Low Threshold Spiking Neurons")
# ax.set_ylim([0, 1.0])
ax.set_xlim([100, 220])
ax.legend()

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/snn_bifurcations.svg')
plt.show()
