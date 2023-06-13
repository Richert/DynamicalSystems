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

path = "results/funcgen"

# load examples
examples = {"s": [], "target": [], "delta": [], "eta": [], "dt": 0.0, "sr": 1,
            "test_prediction": [], "train_prediction": [], "train_phases": [], "test_phases": [],
            "input_indices": [], "K": [], "dim": [], "K_mean": [], "K_var": []
            }
fns = ["SI_funcgen_hom.h5", "SI_funcgen_het.h5"]
for f in fns:
    data = h5py.File(f"{path}/{f}", "r")
    g = data["sweep"]
    examples["delta"].append(np.round(np.asarray(g["Delta"]), decimals=2))
    g = data["data"]
    # examples["eta"].append(np.round(np.asarray(g["eta"]), decimals=0))
    examples["s"].append(np.asarray(g["s"]))
    examples["K"].append(np.asarray(g["K"]))
    examples["target"].append(np.asarray(g["targets"]))
    examples["test_prediction"].append(np.asarray(g["test_predictions"]))
    examples["train_prediction"].append(np.asarray(g["train_predictions"]))
    examples["train_phases"].append(np.asarray(g["train_phases"]))
    examples["test_phases"].append(np.asarray(g["test_phases"]))
    examples["K_mean"].append(np.asarray(g["K_mean"]))
    examples["K_var"].append(np.asarray(g["K_var"]))
    if examples["dt"] == 0.0:
        examples["dt"] = np.asarray(g["dt"])
        examples["sr"] = np.asarray(g["sr"])
        examples["input_indices"] = np.asarray(g["input_indices"])

# load parameter sweep data
res_dict = {"eta": [], "trial": [], "delta": [], "dim": [], "test_loss": [], "phase_variance": [],
            "kernel_variance": []}
for f in os.listdir(path):

    if "snn_funcgen_" in f:

        # load data set
        data = h5py.File(f"{path}/{f}", "r")

        for i in range(len(data)-1):

            # collect simulation data
            g = data[f"{i}"]
            res_dict["eta"].append(np.round(np.asarray(g["eta"]), decimals=0))
            res_dict["dim"].append(np.asarray(g["dimensionality"]))
            test_losses = [mse(np.asarray(g["targets"][1]), np.asarray(sig)) for sig in g["test_predictions"][1]]
            res_dict["test_loss"].append(np.mean(test_losses))
            trial_var = np.asarray(g["kernel_variance"])
            K_var = np.asarray(g["K_var"])
            res_dict["kernel_variance"].append(np.sum(K_var))
            res_dict["phase_variance"].append(trial_var)

            # collect sweep results
            g = data["sweep"]
            res_dict["trial"].append(np.asarray(g["trial"]))
            res_dict["delta"].append(np.round(np.asarray(g["Delta"]), decimals=2))

# turn dictionary into dataframe
res_df = pd.DataFrame.from_dict(res_dict)

# average across trials
#######################

etas_unique = np.unique(res_dict["eta"])
deltas_unique = np.unique(res_dict["delta"])
res_dict_final = {key: [] for key in res_dict.keys() if key not in ["trial"]}
for eta in etas_unique:
    for delta in deltas_unique:
        eta_idx = (res_dict["eta"] - eta) < 1e-3
        delta_idx = (res_dict["delta"] - delta) < 1e-4
        idx = list(np.argwhere(eta_idx*delta_idx > 1e-2).squeeze())
        for key in res_dict_final.keys():
            if key == "trial":
                pass
            elif key == "delta":
                res_dict_final["delta"].append(delta)
            elif key == "eta":
                res_dict_final["eta"].append(eta)
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
# dim = df.pivot(index="eta", columns="delta", values="dim")
sb.lineplot(df, x="delta", y="dim", hue="eta", ax=ax)
ax.set_xlabel(r"$\Delta$")
ax.set_ylabel(r"$d$")
ax.set_title("(A) Dimensionality")
ax.invert_yaxis()

# test loss
ax = fig.add_subplot(grid[0, 1])
sb.lineplot(df, x="delta", y="test_loss", hue="eta", ax=ax)
ax.set_xlabel(r"$\Delta$")
ax.set_ylabel(r"mse")
ax.set_title("(B) MSE (test data)")
ax.invert_yaxis()

# kernel variance
ax = fig.add_subplot(grid[0, 2])
sb.lineplot(df, x="delta", y="kernel_variance", hue="eta", ax=ax)
ax.set_xlabel(r"$\Delta$")
ax.set_ylabel(r"$q$")
ax.set_title("(C) Kernel variation over time")
ax.invert_yaxis()

# kernel variance
# ax = fig.add_subplot(grid[0, 3])
# k = df.pivot(index="alpha", columns="delta", values="phase_variance")
# sb.heatmap(k, cbar=True, ax=ax, xticklabels=ticks, yticklabels=ticks, rasterized=True)
# ax.set_xlabel(r"$\Delta$")
# ax.set_ylabel(r"$\alpha$")
# ax.set_title("(D) Kernel variation over stimulation phases")
# ax.invert_yaxis()

# 1D plots
##########

grid_examples = grid_highlvl[3:].subgridspec(4, 4)

# SNN dynamics
width = int(20.0/(examples["dt"]*examples["sr"]))
indices = examples["input_indices"]
titles = ["E", "F"]
for i, s in enumerate(examples["s"]):
    ax = fig.add_subplot(grid_examples[0, i*2:(i+1)*2])
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
    ax.set_title(fr"({titles[i]}) Network dynamics for all test trials")

# Kernel
grids = [grid_examples[1, 1].subgridspec(1, 1), grid_examples[1, 3].subgridspec(1, 1)]
titles = ["H", "K"]
columns = [1, 3]
xlen = 250
for pc1, title, grid in zip(examples["K_mean"], titles, grids):
    ax = fig.add_subplot(grid[0, 0])
    center = int(len(pc1)/2)
    ax.plot(pc1[center-xlen:center+xlen], color="black")
    ax.set_xlabel("time")
    ax.set_ylabel("")
    ax.set_title(fr"({title}) $K_m$")
grids = [grid_examples[2, 1].subgridspec(1, 1), grid_examples[2, 3].subgridspec(1, 1)]
titles = ["I", "L"]
columns = [1, 3]
ymax = np.max([np.max(p) for p in examples["K_var"]])
for proj, title, grid in zip(examples["K_var"], titles, grids):
    ax = fig.add_subplot(grid[0, 0])
    ax.plot(proj, color="orange")
    ax.set_xlabel("time")
    ax.set_ylabel("")
    ax.set_title(rf"({title}) $K_v$")
    ax.set_ylim([0.0, ymax+0.05*ymax])
grids = [grid_examples[1:3, 0].subgridspec(1, 1), grid_examples[1:3, 2].subgridspec(1, 1)]
titles = ["G", "J"]
columns = [0, 2]
vmax = np.max([np.max(K.flatten()) for K in examples["K"]])
for K, title, grid in zip(examples["K"], titles, grids):
    ax = fig.add_subplot(grid[0, 0])
    sb.heatmap(K, cbar=True, ax=ax, xticklabels=1500, yticklabels=1500, rasterized=True, vmax=vmax, cmap="rocket_r")
    ax.set_xlabel(r"T")
    ax.set_ylabel(r"T")
    ax.set_title(fr"({title}) Network response kernel $K$")

# predictions
grid = grid_examples[3, :].subgridspec(1, 2)
test_example = 4
titles = ["M", "N"]
for i, pred in enumerate(examples["test_prediction"]):
    ax = fig.add_subplot(grid[0, i])
    ax.plot(examples["target"][i][1], label="target", color="black")
    fit = examples["train_prediction"][i][1]
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
plt.savefig(f'results/snn_funcgen.svg')
plt.show()
