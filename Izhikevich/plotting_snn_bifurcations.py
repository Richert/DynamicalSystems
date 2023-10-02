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
snn_results = {key1: {key2: {"folds": [], "hopfs": [], "delta_hopf": [], "delta_fold": []} for key2 in ["gauss", "lorentz"]}
               for key1 in ["rs", "fs", "lts", "rs2"]}
_, _, fnames = next(walk("results/snn_bifurcations"), (None, None, []))
for f in fnames:
    data = pickle.load(open(f"results/snn_bifurcations/{f}", "rb"))
    neuron_type = f.split("_")[1]
    for distribution_type in ["lorentz", "gauss"]:
        if "rs" in neuron_type:
            try:
                folds = data[f"{distribution_type}_fold"]["I_ext"]
                delta = data[f"{distribution_type}_fold"]["delta"] if distribution_type == "lorentz" \
                    else data[f"{distribution_type}_fold"]["sd"]
                snn_results[neuron_type][distribution_type]["folds"].append(folds)
                snn_results[neuron_type][distribution_type]["delta_fold"].append(delta)
            except KeyError:
                pass
        try:
            hopfs = data[f"{distribution_type}_hopf"]["I_ext"]
            delta = data[f"{distribution_type}_hopf"]["delta"] if distribution_type == "lorentz" \
                else data[f"{distribution_type}_hopf"]["sd"]
            snn_results[neuron_type][distribution_type]["hopfs"].append(hopfs)
            snn_results[neuron_type][distribution_type]["delta_hopf"].append(delta)
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
markersize = 1
m = "x"
colors = ["black", "darkorange"]

# create figure
fig = plt.figure(figsize=(6, 5))
grid = fig.add_gridspec(ncols=2, nrows=2)

# plot rs bifurcation diagram for kappa = 10
ax = fig.add_subplot(grid[0, 0])
rs.plot_continuation("PAR(8)", "PAR(5)", cont="D/I:lp1", ignore=["UZ"], line_style_unstable="solid", ax=ax,
                     line_color_stable='#5D6D7E', line_color_unstable='#5D6D7E')
rs.plot_continuation("PAR(8)", "PAR(5)", cont="D/I:lp2", ignore=["UZ"], line_style_unstable="solid", ax=ax,
                     line_color_stable='#5D6D7E', line_color_unstable='#5D6D7E')
ax2 = ax.twinx()
for dist, color, axis in zip(["lorentz", "gauss"], colors, [ax, ax2]):
    fold_left = [fold[0] for fold in snn_results["rs"][dist]["folds"]]
    fold_right = [fold[1] for fold in snn_results["rs"][dist]["folds"]]
    deltas = snn_results["rs"][dist]["delta_fold"]
    axis.scatter(fold_left, deltas, marker=m, color=color, label=dist)
    axis.scatter(fold_right, deltas, marker=m, color=color)
ax.set_xlabel(r"$I_{rs}$ (pA)")
ax.set_ylabel(r"$\Delta_{rs}$ (mV)")
ax2.set_ylabel(r"$\sigma_{rs}$ (mV)")
ax.set_title(r"(A) Regular Spiking Neurons: $\kappa_{rs} = 10$ pA")
ax.set_xlim([0, 80])
ax.set_ylim([0, 3.0])
ax2.set_ylim([0.0, 6.0])
ax.legend()

# plot rs bifurcation diagram for kappa = 100
ax = fig.add_subplot(grid[0, 1])
rs.plot_continuation("PAR(8)", "PAR(5)", cont="D/I:lp3", ignore=["UZ"], line_style_unstable="solid", ax=ax,
                     line_color_stable='#5D6D7E', line_color_unstable='#5D6D7E')
rs.plot_continuation("PAR(8)", "PAR(5)", cont="D/I:lp4", ignore=["UZ"], line_style_unstable="solid", ax=ax,
                     line_color_stable='#5D6D7E', line_color_unstable='#5D6D7E')
rs.plot_continuation("PAR(8)", "PAR(5)", cont="D/I:hb1", ignore=["UZ"], line_style_unstable="solid", ax=ax,
                     line_color_stable='#148F77', line_color_unstable='#148F77')
ax2 = ax.twinx()
for dist, color, axis in zip(["lorentz", "gauss"], colors, [ax, ax2]):
    hopf_left = [fold[0] for fold in snn_results["rs2"][dist]["hopfs"]]
    hopf_right = [fold[1] for fold in snn_results["rs2"][dist]["hopfs"]]
    deltas = snn_results["rs2"][dist]["delta_hopf"]
    axis.scatter(hopf_left, deltas, marker=m, color=color, label=dist)
    axis.scatter(hopf_right, deltas, marker=m, color=color)
ax.set_xlabel(r"$I_{rs}$ (pA)")
ax.set_ylabel(r"$\Delta_{rs}$ (mV)")
ax2.set_ylabel(r"$\sigma_{rs}$ (mV)")
ax.set_title(r"(B) Regular Spiking Neurons: $\kappa_{rs} = 100$ pA")
ax.set_xlim([30, 80])
ax.set_ylim([0, 2.0])
ax2.set_ylim([0.0, 4.0])

# plot fs bifurcation diagram
ax = fig.add_subplot(grid[1, 0])
fs.plot_continuation("PAR(16)", "PAR(6)", cont="D/I:hb1", ignore=["UZ"], line_style_unstable="solid", ax=ax,
                     line_color_stable='#148F77', line_color_unstable='#148F77')
ax2 = ax.twinx()
for dist, color, axis in zip(["lorentz", "gauss"], colors, [ax, ax2]):
    hopf_left = [h[0] for h in snn_results["fs"][dist]["hopfs"]]
    hopf_right = [h[1] for h in snn_results["fs"][dist]["hopfs"]]
    deltas = snn_results["fs"][dist]["delta_hopf"]
    axis.scatter(hopf_left, deltas, marker=m, color=color, label=dist)
    axis.scatter(hopf_right, deltas, marker=m, color=color)
ax.set_xlabel(r"$I_{fs}$ (pA)")
ax.set_ylabel(r"$\Delta_{fs}$ (mV)")
ax2.set_ylabel(r"$\sigma_{fs}$ (mV)")
ax.set_title(r"(C) Fast Spiking Neurons")
ax.set_ylim([0, 0.8])
ax2.set_ylim([0.0, 1.6])
ax.set_xlim([0, 140])
ax.legend()

# plot lts bifurcation diagram
ax = fig.add_subplot(grid[1, 1])
lts.plot_continuation("PAR(16)", "PAR(6)", cont="D/I:hb1", ignore=["UZ"], line_style_unstable="solid", ax=ax,
                      line_color_stable='#148F77', line_color_unstable='#148F77')
ax2 = ax.twinx()
for dist, color, axis in zip(["lorentz", "gauss"], colors, [ax, ax2]):
    hopf_left = [h[0] for h in snn_results["lts"][dist]["hopfs"]]
    hopf_right = [h[1] for h in snn_results["lts"][dist]["hopfs"]]
    deltas = snn_results["lts"][dist]["delta_hopf"]
    axis.scatter(hopf_left, deltas, marker=m, color=color, label=dist)
    axis.scatter(hopf_right, deltas, marker=m, color=color)
ax.set_xlabel(r"$I_{fs}$ (pA)")
ax.set_ylabel(r"$\Delta_{fs}$ (mV)")
ax2.set_ylabel(r"$\sigma_{fs}$ (mV)")
ax.set_title(r"(D) Low Threshold Spiking Neurons")
ax.set_ylim([0, 0.6])
ax2.set_ylim([0.0, 1.2])
ax.set_xlim([100, 220])
ax.legend()

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/snn_bifurcations.svg')
plt.show()
