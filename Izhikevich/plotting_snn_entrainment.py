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
examples = {"s": [], "onsets": [], "pc1": [], "pc1_projection": [], "target": [], "prediction": [], "delta": [],
            "alpha": [], "dt": 0.0, "sr": 1,"input_indices": []}
fns = ["snn_oscillations_hom.h5", "snn_oscillations_het.h5"]
for f in fns:
    data = h5py.File(f"{path}/{f}", "r")
    g = data["sweep"]
    examples["delta"].append(np.round(np.asarray(g["Delta"]), decimals=1))
    g = data["data"]
    examples["alpha"].append(np.round(np.asarray(g["alpha"]), decimals=1))
    examples["s"].append(np.asarray(g["s"]))
    examples["onsets"].append(np.asarray(g["test_onsets"]))
    examples["pc1"].append(np.asarray(g["pcs"])[:, 0])
    examples["pc1_projection"].append(np.asarray(g["pc1_projection"]))
    examples["target"].append(np.asarray(g["targets"]))
    examples["prediction"].append(np.asarray(g["test_predictions"]))
    if examples["dt"] == 0.0:
        examples["dt"] = np.asarray(g["dt"])
        examples["sr"] = np.asarray(g["sr"])
        examples["input_indices"] = np.asarray(g["input_indices"])

# load parameter sweep data
res_dict = {"alpha": [], "trial": [], "delta": [], "dim": [], "test_loss": [], "kernel_distortion": []}
for f in os.listdir(path):

    if "snn_oscillatory_" in f:

        # load data set
        data = h5py.File(f"{path}/{f}", "r")

        for i in range(len(data)-1):

            # collect simulation data
            g = data[f"{i}"]
            res_dict["alpha"].append(np.round(np.asarray(g["alpha"]), decimals=1))
            res_dict["dim"].append(np.asarray(g["dimensionality"]))
            test_losses = [mse(np.asarray(g["targets"][1]), np.asarray(sig)) for sig in g["test_predictions"][1]]
            res_dict["test_loss"].append(np.mean(test_losses))
            pc1_proj = np.real(np.asarray(g["pc1_projection"]))
            pc1 = np.real(np.asarray(g["pcs"][:, 0]))
            pc1 -= np.min(pc1)
            pc1 /= np.max(pc1)
            peaks, _ = find_peaks(pc1, height=0.05, width=5)
            res_dict["kernel_distortion"].append(len(peaks)*np.var(pc1_proj))

            # collect sweep results
            g = data["sweep"]
            res_dict["trial"].append(np.asarray(g["trial"]))
            res_dict["delta"].append(np.round(np.asarray(g["Delta"]), decimals=2))

# turn dictionary into dataframe
res_df = pd.DataFrame.from_dict(res_dict)

# average across trials
#######################

alphas_unique = np.unique(res_dict["alpha"])
deltas_unique = np.unique(res_dict["delta"])
res_dict_final = {key: [] for key in res_dict.keys()}
for alpha in alphas_unique:
    for delta in deltas_unique:
        alpha_idx = (res_dict["alpha"] - alpha) < 1e-3
        delta_idx = (res_dict["delta"] - delta) < 1e-4
        idx = list(np.argwhere(alpha_idx*delta_idx > 1e-2).squeeze())
        for key in res_dict_final.keys():
            if key == "alpha":
                res_dict_final["alpha"].append(alpha)
            elif key == "delta":
                res_dict_final["delta"].append(delta)
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
ticks = 5

# create figure layout
fig = plt.figure(figsize=(12, 8), constrained_layout=True)
grid_highlvl = fig.add_gridspec(5, 1)

# 2D plots
##########

grid = grid_highlvl[:2].subgridspec(1, 3)

# test loss
ax = fig.add_subplot(grid[0, 0])
test_loss = df.pivot(index="alpha", columns="delta", values="test_loss")
sb.heatmap(test_loss, cbar=True, ax=ax, xticklabels=ticks, yticklabels=ticks, rasterized=True)
ax.set_xlabel(r"$\Delta$")
ax.set_ylabel(r"$\alpha$")
ax.set_title("MSE (test data)")

# dimensionality
ax = fig.add_subplot(grid[0, 1])
dim = df.pivot(index="alpha", columns="delta", values="dim")
sb.heatmap(dim, cbar=True, ax=ax, xticklabels=ticks, yticklabels=ticks, rasterized=True)
ax.set_xlabel(r"$\Delta$")
ax.set_ylabel(r"$\alpha$")
ax.set_title("Dimensionality")

# kernel variance
ax = fig.add_subplot(grid[0, 2])
k = df.pivot(index="alpha", columns="delta", values="kernel_distortion")
sb.heatmap(k, cbar=True, ax=ax, xticklabels=ticks, yticklabels=ticks, rasterized=True)
ax.set_xlabel(r"$\Delta$")
ax.set_ylabel(r"$\alpha$")
ax.set_title("Kernel distortion")

# 1D plots
##########

grid = grid_highlvl[2:].subgridspec(3, 2)

# SNN dynamics
width = int(20.0/(examples["dt"]*examples["sr"]))
indices = examples["input_indices"]
for i, s in enumerate(examples["s"]):
    ax = fig.add_subplot(grid[0, i])
    s_all = np.concatenate(s, axis=1)
    s_all /= np.max(s_all)
    im = ax.imshow(s_all, aspect="auto", interpolation="none", cmap="Greys")
    plt.sca(ax)
    dur = 0
    for n in range(len(s)):
        plt.fill_betweenx(y=indices, x1=[dur for _ in range(len(indices))],
                          x2=[width + dur for _ in range(len(indices))], color='red', alpha=0.5)
        dur += len(s[n, 0])
        ax.axvline(x=dur, color="blue", linestyle="solid")
    ax.set_xlabel('time')
    ax.set_ylabel('neurons')
    ax.set_title(fr"$\alpha = {examples['alpha'][i]}$, $\Delta = {examples['delta'][i]}$")

# PC1
for i, pc1 in enumerate(examples["pc1"]):
    ax = fig.add_subplot(grid[1, i])
    pc1_proj = examples["pc1_projection"][i]
    if i == 0:
        ax2 = inset_axes(ax, width="30%", height="50%", loc=1)
        ax2.plot(pc1, color="black", label="PC1")
        ax2.legend()
        ax2.set_yticks([])
        ax2.set_xticks([])
    ax.plot(pc1_proj, color="orange")
    ax.set_xlabel("time")
    ax.set_ylabel("")
    ax.set_title(r"Projection onto 1. PC of $K$")
    ax.set_ylim([0.0, 0.1])

# predictions
example = 0
for i, pred in enumerate(examples["prediction"]):
    ax = fig.add_subplot(grid[2, i])
    ax.plot(examples["target"][i][1], label="target", color="black")
    ax.plot(pred[1][example], label="prediction", color="orange")
    if i == 0:
        ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel("")
    ax.set_title(f"Function generation for test trial {example+1}")

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/snn_oscillations.svg')
plt.show()
