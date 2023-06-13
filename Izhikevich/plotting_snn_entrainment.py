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

path = "results/oscillatory"

# load examples
examples = {"s": [], "train_phases": [], "test_phases": [], "train_predictions": [], "test_predictions": [],
            "targets": [],  "delta": [], "alpha": [], "dt": 0.0, "sr": 1, "input_indices": [],
            "K": [], "K_mean": [], "K_var": [], "K_diag": []
            }
fns = ["SI_oscillations_hom.h5", "SI_oscillations_het.h5"]
for f in fns:
    data = h5py.File(f"{path}/{f}", "r")
    g = data["sweep"]
    examples["delta"].append(np.round(np.asarray(g["Delta"]), decimals=1))
    g = data["data"]
    examples["alpha"].append(np.round(np.asarray(g["alpha"]), decimals=1))
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

# load parameter sweep data
res_dict = {"alpha": [], "trial": [], "delta": [], "dim": [], "test_loss": [], "kernel_quality": [], "test_phase": []}
for f in os.listdir(path):

    if "snn_oscillatory_" in f:

        # load data set
        data = h5py.File(f"{path}/{f}", "r")

        # extract data
        g = data[f"data"]
        test_losses = [mse(np.asarray(g["targets"][1]), np.asarray(sig)) for sig in g["test_predictions"][1]]
        test_phases = np.round(np.asarray(g["test_phases"]), decimals=2)
        k_mean = np.asarray(g["K_mean"])
        k_var = np.asarray(g["K_var"])
        alpha = np.round(np.asarray(g["alpha"]), decimals=1)
        dim = np.asarray(g["dimensionality"])
        g = data["sweep"]
        trial = np.asarray(g["trial"])
        delta = np.round(np.asarray(g["Delta"]), decimals=2)

        for loss, phase in zip(test_losses, test_phases):

            # calculate kernel quality
            kernel_quality = np.sum(k_var)

            # collect simulation data
            res_dict["alpha"].append(alpha)
            res_dict["dim"].append(dim)
            res_dict["test_loss"].append(loss)
            res_dict["test_phase"].append(phase)
            res_dict["kernel_quality"].append(kernel_quality)

            # collect sweep results
            res_dict["trial"].append(trial)
            res_dict["delta"].append(delta)

# turn dictionary into dataframe
res_df = pd.DataFrame.from_dict(res_dict)

# average across trials
#######################

deltas_unique = np.unique(res_dict["delta"])
phases_unique = np.unique(res_dict["test_phase"])
res_dict_final = {key: [] for key in res_dict.keys() if key not in ["alpha", "trial"]}
for delta in deltas_unique:
    for phase in phases_unique:
        delta_idx = np.abs(res_dict["delta"] - delta) < 1e-4
        phase_idx = np.abs(res_dict["test_phase"] - phase) < 1e-3
        idx = list(np.argwhere(delta_idx*phase_idx > 1e-2).squeeze())
        for key in res_dict_final.keys():
            if key == "alpha" or key == "trial":
                pass
            elif key == "delta":
                res_dict_final["delta"].append(delta)
            elif key == "test_phase":
                res_dict_final["test_phase"].append(phase)
            else:
                res_dict_final[key].append(np.mean(res_df.loc[idx, key].values))

df = pd.DataFrame.from_dict(res_dict_final)

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

# create figure layout
fig = plt.figure(figsize=(12, 9), constrained_layout=True)
grid_highlvl = fig.add_gridspec(10, 1)

# 2D plots
##########

grid = grid_highlvl[:3].subgridspec(1, 4)

# dimensionality
ax = fig.add_subplot(grid[0, 0])
dim = df.pivot(index="delta", columns="test_phase", values="dim").mean(axis=1)
ax.plot(dim)
ax.set_ylabel("$d$")
ax.set_xlabel(r"$\Delta$")
ax.set_title("(A) Dimensionality")

# test loss
ax = fig.add_subplot(grid[0, 1])
test_loss = df.pivot(index="delta", columns="test_phase", values="test_loss")
sb.heatmap(test_loss, cbar=True, ax=ax, xticklabels=ticks, yticklabels=ticks, rasterized=True, vmax=0.3)
ax.set_xlabel(r"phase")
ax.set_ylabel(r"$\Delta$")
ax.set_title("(B) MSE (test data)")
ax.invert_yaxis()

# kernel variance
ax = fig.add_subplot(grid[0, 2])
k = df.pivot(index="delta", columns="test_phase", values="kernel_quality").mean(axis=1)
ax.plot(k)
ax.set_xlabel(r"$q$")
ax.set_xlabel(r"$\Delta$")
ax.set_title("(C) Kernel quality")

# kernel variance
ax = fig.add_subplot(grid[0, 3])
mean_loss = df.pivot(index="delta", columns="test_phase", values="test_loss").mean(axis=1)
ax.plot(mean_loss)
ax.set_xlabel(r"$\Delta$")
ax.set_ylabel(r"MSE")
ax.set_title("(D) Average test loss")

# 1D plots
##########

grid_examples = grid_highlvl[3:].subgridspec(4, 4)

# SNN dynamics
width = int(20.0/(examples["dt"]*examples["sr"]))
indices = examples["input_indices"]
titles = ["E", "F"]
Cs = []
for i, s in enumerate(examples["s"]):
    ax = fig.add_subplot(grid_examples[0, i*2:(i+1)*2])
    s_tmp = s[np.arange(0, len(s), 3)]
    s_all = np.concatenate(s_tmp, axis=1)
    s_all /= np.max(s_all)
    C = np.corrcoef(s_all)
    C[np.isnan(C)] = 0.0
    Cs.append(np.abs(C))
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
    ax.set_title(fr"({titles[i]}) Network dynamics on test trials")

# Kernel
grids = [grid_examples[1:3, 0].subgridspec(1, 1), grid_examples[1:3, 2].subgridspec(1, 1)]
titles = ["G", "I"]
vmax = np.max([np.max(K.flatten()) for K in examples["K"]])
for K, title, grid in zip(examples["K"], titles, grids):
    ax = fig.add_subplot(grid[0, 0])
    sb.heatmap(K, cbar=True, ax=ax, xticklabels=1500, yticklabels=1500, rasterized=True, vmax=vmax, cmap="rocket_r")
    ax.set_xlabel(r"T")
    ax.set_ylabel(r"T")
    ax.set_title(fr"({title}) Network response kernel $K$")
grids = [grid_examples[1:3, 1].subgridspec(1, 1), grid_examples[1:3, 3].subgridspec(1, 1)]
titles = ["H", "J"]
vmax = np.max([np.max(C.flatten()) for C in Cs])
for C, title, grid in zip(Cs, titles, grids):
    ax = fig.add_subplot(grid[0, 0])
    sb.heatmap(C, cbar=True, ax=ax, xticklabels=1500, yticklabels=1500, rasterized=True, vmax=vmax)
    ax.set_xlabel(r"N")
    ax.set_ylabel(r"N")
    ax.set_title(fr"({title}) Neural correlations $C$")

# predictions
grid = grid_examples[3, :].subgridspec(1, 2)
test_example = 4
titles = ["K", "L"]
for i, pred in enumerate(examples["test_predictions"]):
    ax = fig.add_subplot(grid[0, i])
    ax.plot(examples["targets"][i][1], label="target", color="black")
    fit = examples["train_predictions"][i][1]
    ax.plot(fit, label="fit", color="blue")
    ax.plot(pred[1][test_example], label="prediction", color="orange")
    if i == 1:
        ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel("")
    ax.set_title(f"({titles[i]}) Function generation performance")

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/snn_oscillations.svg')
plt.show()
