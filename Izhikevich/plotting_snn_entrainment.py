import os
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import h5py
from matplotlib.gridspec import GridSpec
from scipy.signal import find_peaks
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def mse(x: np.ndarray, y: np.ndarray) -> float:
    x = x.squeeze() / np.max(x.flatten())
    y = y.squeeze() / np.max(y.flatten())
    return float(np.mean((x - y)**2))


# load data
###########

# load examples
examples = {"s": [], "train_phases": [], "test_phases": [], "train_predictions": [], "test_predictions": [],
            "targets": [],  "delta": [], "dt": 0.0, "sr": 1, "input_indices": [],
            "K": [], "K_mean": [], "K_var": [], "K_diag": []
            }
fns = [
    #"results/oscillatory/SI_oscillations_hom.h5", "results/oscillatory/SI_oscillations_het.h5",
    #"results/funcgen/SI_async_low_hom.h5", "results/funcgen/SI_async_low_het.h5",
    "results/funcgen/SI_async_high_hom.h5", "results/funcgen/SI_async_high_het.h5"
       ]
for f in fns:
    data = h5py.File(f, "r")
    g = data["sweep"]
    examples["delta"].append(np.round(np.asarray(g["Delta"]), decimals=1))
    g = data["data"]
    examples["s"].append(np.asarray(g["s"]))
    examples["train_phases"].append(np.round(np.asarray(g["train_phases"]), decimals=2))
    examples["test_phases"].append(np.round(np.asarray(g["test_phases"]), decimals=2))
    examples["targets"].append(np.asarray(g["targets"]))
    examples["train_predictions"].append(np.asarray(g["train_predictions"]))
    examples["test_predictions"].append(np.asarray(g["test_predictions"]))
    examples["K"].append(np.asarray(g["K"]))
    examples["K_mean"].append(np.asarray(g["K_mean"]))
    examples["K_var"].append(np.asarray(g["K_var"]))
    examples["K_diag"].append(np.asarray(g["K_diag"]))
    if examples["dt"] == 0.0:
        examples["dt"] = np.asarray(g["dt"])
        examples["sr"] = np.asarray(g["sr"])
        examples["input_indices"] = np.asarray(g["input_indices"])

############
# plotting #
############

# plot settings
plt.switch_backend("TkAgg")
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
ticks = 6

# create figure layout
fig = plt.figure(figsize=(12, 9), constrained_layout=True)

# 1D plots
##########

grid_examples = fig.add_gridspec(5, 4)

# SNN dynamics
width = int(20.0/(examples["dt"]*examples["sr"]))
indices = examples["input_indices"]
titles = ["A", "B"]
delta_str = "\Delta_{rs}"
Cs, CVs = [], []
for i, s in enumerate(examples["s"]):
    ax = fig.add_subplot(grid_examples[0, i*2:(i+1)*2])
    s_tmp = s[np.arange(0, len(s), 3)]
    s_all = np.concatenate(s_tmp, axis=1)
    s_all /= np.max(s_all)
    Cs_tmp = []
    for signal in s_tmp:
        C = np.corrcoef(signal)
        C[np.isnan(C)] = 0.0
        Cs_tmp.append(C)
    Cs.append(np.abs(np.mean(Cs_tmp, axis=0)))
    CVs.append(np.var(Cs_tmp, axis=0))
    phases = np.round(np.mod(np.arange(0, s_all.shape[1]), s[0].shape[1]) * np.pi * 2.0 / s[0].shape[1], decimals=2)
    phase_ticks = np.arange(0, len(phases), 1190)
    im = ax.imshow(s_all, aspect="auto", interpolation="none", cmap="Greys")
    plt.sca(ax)
    dur = 0
    for n in range(len(s_tmp)):
        plt.fill_betweenx(y=indices, x1=[dur for _ in range(len(indices))],
                          x2=[width + dur for _ in range(len(indices))], color='red', alpha=0.5)
        dur += len(s_tmp[n, 0])
        ax.axvline(x=dur, color="blue", linestyle="solid")
    ax.set_xticks(phase_ticks, labels=phases[phase_ticks])
    ax.set_xlabel('phase')
    ax.set_ylabel('neurons')
    ax.set_title(fr"({titles[i]}) Network dynamics (${delta_str} = {examples['delta'][i]}$ mV)")

# Kernel
grids = [grid_examples[1:3, 0].subgridspec(1, 1), grid_examples[1:3, 2].subgridspec(1, 1)]
titles = ["C", "E"]
for C, title, grid in zip(Cs, titles, grids):
    ax = fig.add_subplot(grid[0, 0])
    sb.heatmap(C, cbar=True, ax=ax, xticklabels=300, yticklabels=300, rasterized=True, vmax=1, vmin=0)
    ax.set_xlabel(r"N")
    ax.set_ylabel(r"N")
    ax.set_title(fr"({title}) Neural correlations")
grids = [grid_examples[1:3, 1].subgridspec(1, 1), grid_examples[1:3, 3].subgridspec(1, 1)]
titles = ["D", "F"]
vmax = np.max([np.max(C.flatten()) for C in CVs])
vmin = np.min([np.min(C.flatten()) for C in CVs])
for C, title, grid in zip(CVs, titles, grids):
    ax = fig.add_subplot(grid[0, 0])
    sb.heatmap(C, cbar=True, ax=ax, xticklabels=300, yticklabels=300, rasterized=True, vmax=vmax, vmin=vmin)
    ax.set_xlabel(r"N")
    ax.set_ylabel(r"N")
    ax.set_title(fr"({title}) Variance across trials")

# predictions
grid = grid_examples[3:, :].subgridspec(2, 2)
test_example = 1
titles = ["G", "H"]
for i, pred in enumerate(examples["test_predictions"]):
    ax = fig.add_subplot(grid[0, i])
    target = examples["targets"][i][1]
    ax.plot(target, label="target", color="black")
    fit = examples["train_predictions"][i][1]
    ax.plot(fit, label="fit", color="blue")
    ax.plot(pred[1][test_example], label="prediction", color="orange")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(fr"MSE = {mse(target, pred[1][test_example])}")
    ax = fig.add_subplot(grid[1, i])
    target = examples["targets"][i][0]
    ax.plot(examples["targets"][i][0], label="target", color="black")
    fit = examples["train_predictions"][i][0]
    ax.plot(fit, label="fit", color="blue")
    ax.plot(pred[0][test_example], label="prediction", color="orange")
    if i == 1:
        ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel("")
    ax.set_title(fr"MSE = {mse(target, pred[0][test_example])}")

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/SI_asynch_high.svg')
plt.show()
