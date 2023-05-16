import os
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import h5py
from matplotlib.gridspec import GridSpec


def mse(x: np.ndarray, y: np.ndarray) -> float:
    x = x.squeeze() / np.max(x.flatten())
    y = y.squeeze() / np.max(y.flatten())
    return float(np.mean((x - y)**2))


# load data
###########

res_dict = {"alpha": [], "dim": [], "seq": [], "train_loss_1": [], "train_loss_2": [], "test_loss_1": [],
            "test_loss_2": [], "trial": [], "delta": [], "kernel_width": [], "kernel_variance": []}

path = "results/entrainment"
for f in os.listdir(path):

    # load data set
    data = h5py.File(f"{path}/{f}", "r")

    for i in range(len(data)-1):

        # collect simulation data
        g = data[f"{i}"]
        res_dict["alpha"].append(np.round(np.asarray(g["alpha"]), decimals=1))
        res_dict["dim"].append(np.asarray(g["dimensionality"]))
        res_dict["seq"].append(np.asarray(g["sequentiality"]))
        res_dict["train_loss_1"].append(mse(np.asarray(g["targets"][0]), np.asarray(g["train_predictions"][0])))
        res_dict["train_loss_2"].append(mse(np.asarray(g["targets"][1]), np.asarray(g["train_predictions"][1])))
        test_losses = [mse(np.asarray(g["targets"][0]), np.asarray(sig)) for sig in g["test_predictions"][0]]
        res_dict["test_loss_1"].append(np.mean(test_losses))
        test_losses = [mse(np.asarray(g["targets"][1]), np.asarray(sig)) for sig in g["test_predictions"][1]]
        res_dict["test_loss_2"].append(np.mean(test_losses))
        pc1 = np.asarray(g["pcs"])[:, 0]
        hm = np.max(pc1)*0.5
        idx0 = np.argmin(pc1[:int(0.5*len(pc1))] - hm).squeeze()
        idx1 = np.argmin(pc1[int(0.5 * len(pc1)):] - hm).squeeze()
        res_dict["kernel_width"].append(idx1 - idx0)
        pc1_proj = np.asarray(g["pc1_projection"])
        res_dict["kernel_variance"].append(np.var(pc1_proj))

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

# plotting
##########

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6

# plotting parameters
ticks = 3
# create figure layout
fig = plt.figure(1)
grid = GridSpec(nrows=2, ncols=3, figure=fig)

# dimensionality
ax = fig.add_subplot(grid[0, 0])
dim = df.pivot(index="alpha", columns="delta", values="dim")
sb.heatmap(dim, cbar=True, ax=ax, xticklabels=ticks, yticklabels=ticks, rasterized=True)
ax.set_xlabel(r"$\Delta$")
ax.set_ylabel(r"$\alpha$")
ax.set_title("Dimensionality")

# dimensionality
ax = fig.add_subplot(grid[1, 0])
seq = df.pivot(index="alpha", columns="delta", values="seq")
sb.heatmap(seq, cbar=True, ax=ax, xticklabels=ticks, yticklabels=ticks, rasterized=True)
ax.set_xlabel(r"$\Delta$")
ax.set_ylabel(r"$\alpha$")
ax.set_title("Sequentiality")

# train loss 1
# ax = fig.add_subplot(grid[0, 1])
# train_loss = df.pivot(index="alpha", columns="delta", values="train_loss_1")
# sb.heatmap(train_loss, cbar=True, ax=ax, xticklabels=ticks, yticklabels=ticks, rasterized=True)
# ax.set_xlabel(r"$\Delta$")
# ax.set_ylabel(r"$\alpha$")
# ax.set_title("Task I: Train loss")
#
# # train loss 2
# ax = fig.add_subplot(grid[1, 1])
# train_loss = df.pivot(index="alpha", columns="delta", values="train_loss_2")
# sb.heatmap(train_loss, cbar=True, ax=ax, xticklabels=ticks, yticklabels=ticks, rasterized=True)
# ax.set_xlabel(r"$\Delta$")
# ax.set_ylabel(r"$\alpha$")
# ax.set_title("Task II: Train loss")

# test loss 1
ax = fig.add_subplot(grid[0, 1])
test_loss = df.pivot(index="alpha", columns="delta", values="test_loss_1")
sb.heatmap(test_loss, cbar=True, ax=ax, xticklabels=ticks, yticklabels=ticks, rasterized=True)
ax.set_xlabel(r"$\Delta$")
ax.set_ylabel(r"$\alpha$")
ax.set_title("Task I: Test loss")

# train loss 2
ax = fig.add_subplot(grid[1, 1])
test_loss = df.pivot(index="alpha", columns="delta", values="test_loss_2")
sb.heatmap(test_loss, cbar=True, ax=ax, xticklabels=ticks, yticklabels=ticks, rasterized=True)
ax.set_xlabel(r"$\Delta$")
ax.set_ylabel(r"$\alpha$")
ax.set_title("Task II: Test loss")

# kernel width
ax = fig.add_subplot(grid[0, 2])
kwidth = df.pivot(index="alpha", columns="delta", values="kernel_width")
sb.heatmap(kwidth, cbar=True, ax=ax, xticklabels=ticks, yticklabels=ticks, rasterized=True)
ax.set_xlabel(r"$\Delta$")
ax.set_ylabel(r"$\alpha$")
ax.set_title("Kernel width")

# kernel variance
ax = fig.add_subplot(grid[1, 2])
kvar = df.pivot(index="alpha", columns="delta", values="kernel_variance")
sb.heatmap(kvar, cbar=True, ax=ax, xticklabels=ticks, yticklabels=ticks, rasterized=True)
ax.set_xlabel(r"$\Delta$")
ax.set_ylabel(r"$\alpha$")
ax.set_title("Kernel variance")

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/snn_oscillations.svg')
plt.show()
