import os
import matplotlib
matplotlib.use('tkagg')
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import h5py


def mse(x: np.ndarray, y: np.ndarray) -> float:
    x = x.squeeze() / np.max(x.flatten())
    y = y.squeeze() / np.max(y.flatten())
    return float(np.mean((x - y)**2))


# load data
###########

cond = "oscillations"
res_dir = "oscillatory"

# load examples
examples = {"s": [], "train_phases": [], "test_phases": [], "train_predictions": [], "test_predictions": [],
            "targets": [],  "delta": [], "dt": 0.0, "sr": 1, "input_indices": [], "G": [], "dim": [], "K": [],
            }
fns = [f"results/{res_dir}/SI_{cond}_hom.h5", f"results/{res_dir}/SI_{cond}_het.h5"]
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
    examples["G"].append(np.asarray(g["G"]))
    examples["K"].append(np.asarray(g["K"]))
    examples["dim"].append(g["dimensionality"][()])
    if examples["dt"] == 0.0:
        examples["dt"] = np.asarray(g["dt"])
        examples["sr"] = np.asarray(g["sr"])
        examples["input_indices"] = np.asarray(g["input_indices"])

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
ticks = 6
color = cmap = sb.crayon_palette(["Indigo"])[0]
# create figure layout
fig = plt.figure(figsize=(12, 9), constrained_layout=True)

# 1D plots
##########

grid_examples = fig.add_gridspec(9, 4)

# SNN dynamics
width = int(20.0/(examples["dt"]*examples["sr"]))
indices = examples["input_indices"]
titles = ["A", "B"]
delta_str = "\Delta_{rs}"
Cs = []
for i, s in enumerate(examples["s"]):
    ax = fig.add_subplot(grid_examples[:2, i*2:(i+1)*2])
    s_tmp = s[np.arange(0, len(s), 3)]
    s_all = np.concatenate(s_tmp, axis=1)
    s_all /= np.max(s_all)
    Cs_tmp = []
    for signal in s:
        C = np.corrcoef(signal)
        idx = np.sum(signal, axis=1) < 1e-6
        C[idx, :] = 0.0
        C[:, idx] = 0.0
        Cs_tmp.append(C)
    Cs.append(np.mean(Cs_tmp, axis=0))
    phases = np.round(np.mod(np.arange(0, s_all.shape[1]), s[0].shape[1]) * np.pi * 2.0 / s[0].shape[1], decimals=2)
    phases[phases == 6.28] = 0.0
    phase_ticks = np.arange(0, len(phases), int(s[0].shape[-1]/2))
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
grid_kernels = grid_examples[2:5, :].subgridspec(1, 4)
grids = [grid_kernels[0, 0], grid_kernels[0, 2]]
titles = ["C", "E"]
for C, title, grid, dim in zip(Cs, titles, grids, examples["dim"]):
    ax = fig.add_subplot(grid)
    sb.heatmap(C, cbar=True, ax=ax, xticklabels=300, yticklabels=300, rasterized=True, vmax=1, vmin=-1, cmap="icefire")
    ax.set_xlabel(r"neuron ID")
    ax.set_ylabel(r"neuron ID")
    ax.set_title(fr"({title}) Neural correlations ($d = {np.round(dim, decimals=1)}$)")
    ax.invert_yaxis()
grids = [grid_kernels[0, 1], grid_kernels[0, 3]]
titles = ["D", "F"]
for K, G, title, grid in zip(examples["K"], examples["G"], titles, grids):
    ax = fig.add_subplot(grid)
    sb.heatmap(K, cbar=True, ax=ax, xticklabels=600, yticklabels=600, rasterized=True, cmap="icefire")
    ax.set_xlabel(r"time step")
    ax.set_ylabel(r"time step")
    q = np.log(np.sum(np.abs(G.flatten())))
    ax.set_title(fr"({title}) Network response kernel ($\log(q) = {np.round(q, decimals=1)}$)")

# predictions
grid = grid_examples[5:, :].subgridspec(2, 2)
test_example = 1
titles = ["G", "H"]
for i, pred in enumerate(examples["test_predictions"]):
    ax = fig.add_subplot(grid[0, i])
    target = examples["targets"][i][1]
    ax.plot(target, label="target", color="black")
    fit = examples["train_predictions"][i][1]
    ax.plot(fit, label="fit", color=color)
    ax.plot(pred[1][test_example], label="prediction", color="orange")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(fr"MSE = {np.round(mse(target, pred[1][test_example]), decimals=2)}")
    ax = fig.add_subplot(grid[1, i])
    target = examples["targets"][i][0]
    tmax = np.max(target)
    ax.plot(target / tmax, label="target", color="black")
    fit = examples["train_predictions"][i][0]
    ax.plot(fit / tmax, label="fit", color=color)
    ax.plot(pred[0][test_example] / tmax, label="prediction", color="orange")
    if i == 1:
        ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel("")
    ax.set_title(fr"MSE = {np.round(mse(target / tmax, pred[0][test_example] / tmax), decimals=2)}")

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/SI_{cond}.svg')
plt.show()
