import os
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

cond = "hom"
path = "results/oscillatory"

# load examples
examples = {}
f = f"SI_oscillations_{cond}.h5"
data = h5py.File(f"{path}/{f}", "r")
g = data["sweep"]
examples["delta"] = np.round(np.asarray(g["Delta"]), decimals=1)
g = data["data"]
examples["alpha"] = np.round(np.asarray(g["alpha"]), decimals=1)
examples["s"] = np.asarray(g["s"])
examples["target"] = np.asarray(g["targets"])
examples["test_prediction"] = np.asarray(g["test_predictions"])
examples["train_prediction"] = np.asarray(g["train_predictions"])
# examples["train_phases"] = np.asarray(g["train_phases"])
# examples["test_phases"] = np.asarray(g["test_phases"])
examples["dt"] = np.asarray(g["dt"])
examples["sr"] = np.asarray(g["sr"])
examples["input_indices"] = np.asarray(g["input_indices"])
examples["dim"] = np.round(np.asarray(g["dimensionality"]), decimals=1)
examples["K"] = np.asarray(g["K"])
examples["K_mean"] = np.asarray(g["K_mean"])
examples["K_var"] = np.asarray(g["K_var"])
examples["K_diag"] = np.asarray(g["K_diag"])

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
ticks = 5

# create figure layout
fig = plt.figure(figsize=(12, 7), constrained_layout=True)
grid_highlvl = fig.add_gridspec(3, 2)

# SNN dynamics
width = int(20.0/(examples["dt"]*examples["sr"]))
indices = examples["input_indices"]
ax = fig.add_subplot(grid_highlvl[0, 0])
s = examples["s"][np.arange(0, len(examples["s"]), 3)]
s_all = np.concatenate(s, axis=1)
s_all /= np.max(s_all)
phases = np.round(np.mod(np.arange(0, s_all.shape[1]), s[0].shape[1])*np.pi*2.0/s[0].shape[1], decimals=2)
phase_ticks = np.arange(0, len(phases), 1190)
im = ax.imshow(s_all, aspect="auto", interpolation="none", cmap="Greys")
plt.sca(ax)
dur = 0
for n in range(len(s)):
    plt.fill_betweenx(y=indices, x1=[dur for _ in range(len(indices))],
                      x2=[width + dur for _ in range(len(indices))], color='red', alpha=0.5)
    dur += len(s[n, 0])
    ax.axvline(x=dur, color="blue", linestyle="solid")
ax.set_xticks(phase_ticks, labels=phases[phase_ticks])
ax.set_xlabel('phase')
ax.set_ylabel('neurons')
ax.set_title(fr"(A) Network dynamics for all test trials")

# covariance and kernel matrices
grid = grid_highlvl[1:, 0].subgridspec(1, 1)
# ax = fig.add_subplot(grid[0, 0])
# sb.heatmap(examples["K"], cbar=True, ax=ax, xticklabels=1500, yticklabels=1500, rasterized=True, cmap="magma_r")
# ax.set_xlabel(r"T")
# ax.set_ylabel(r"T")
# ax.set_title(fr"(B) Network response kernel $K$")
ax = fig.add_subplot(grid[0, 0])
dim = examples["dim"]
C = np.corrcoef(s_all)
C[np.isnan(C)] = 0.0
sb.heatmap(np.abs(C), cbar=True, ax=ax, xticklabels=400, yticklabels=400, rasterized=True, vmin=0, vmax=1)
ax.set_xlabel(r"N")
ax.set_ylabel(r"N")
ax.set_title(fr"(B) Neural correlations (dimensionality = {dim})")

# PC1 and PC1 projection
# grid = grid_highlvl[2, 0].subgridspec(1, 2)
# xlen = 250
# ax = fig.add_subplot(grid[0, 0])
# K_mean = examples["K_mean"]
# K_var = examples["K_var"]
# center = int(len(K_mean)/2)
# K_mean = K_mean[center-xlen:center+xlen]
# K_var = K_var[center-xlen:center+xlen]
# ax.plot(K_mean, color="black")
# ax.set_xlabel("time")
# ax.set_ylabel("")
# ax.set_ylim([0.0, 0.009])
# ax.set_title(r"(D) $K_{mean}$")
# ax = fig.add_subplot(grid[0, 1])
# ax.plot(K_var, color="orange")
# ax.set_xlabel("time")
# ax.set_ylabel("")
# ax.set_ylim([0.0, 7e-6])
# ax.set_title(r"(E) $K_{var}$")

# predictions
test_examples = [0, 6]
titles = ["F", "G"]
row = 0
grid = grid_highlvl[:, 1].subgridspec(4, 1)
for ex, title in zip(test_examples, titles):

    # target 1
    ax = fig.add_subplot(grid[row, 0])
    ax.plot(examples["target"][1], label="target", color="black")
    ax.plot(examples["train_prediction"][1], label="fit", color="blue")
    ax.plot(examples["test_prediction"][1][ex], label="prediction", color="orange")
    if row == 0:
        ax.set_title(f"(C) Function generation performance on target 1")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_ylim([-1.2, 1.2])

    # target 2
    ax = fig.add_subplot(grid[row+2, 0])
    tmax = np.max(examples["target"][0])
    ax.plot(examples["target"][0] / tmax, label="target", color="black")
    ax.plot(examples["train_prediction"][0] / tmax, label="fit", color="blue")
    ax.plot(examples["test_prediction"][0][ex] / tmax, label="prediction", color="orange")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_ylim([-0.2, 1.2])
    if row == 0:
        ax.set_title(f"(D) Function generation performance on target 2")
    if row == 1:
        ax.legend()
        ax.set_xlabel("time")
    row += 1

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/SI_oscillations_{cond}.svg')
plt.show()
