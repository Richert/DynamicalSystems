import os
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import h5py
from pycobi import ODESystem


def mse(x: np.ndarray, y: np.ndarray) -> float:
    x = x.squeeze() / np.max(x.flatten())
    y = y.squeeze() / np.max(y.flatten())
    return float(np.mean((x - y)**2))


# load data
###########

# load data from asynchronous regimes
path = "results/funcgen"
res_dict = {"eta": [], "trial": [], "delta": [], "dim": [], "test_loss_1": [], "test_loss_2": [], "phase_variance": [],
            "kernel_variance": [], "test_var_1": [], "test_var_2": [],}
for f in os.listdir(path):

    if "snn_funcgen_" in f:

        # load data set
        data = h5py.File(f"{path}/{f}", "r")

        for i in range(len(data)-1):

            # collect simulation data
            g = data[f"{i}"]
            res_dict["eta"].append(np.round(g["eta"][()], decimals=0))
            res_dict["dim"].append(g["dimensionality"][()])
            test_losses_1 = [mse(np.asarray(g["targets"][0]), np.asarray(sig)) for sig in g["test_predictions"][0]]
            res_dict["test_loss_1"].append(np.mean(test_losses_1))
            res_dict["test_var_1"].append(np.var(test_losses_1))
            test_losses_2 = [mse(np.asarray(g["targets"][1]), np.asarray(sig)) for sig in g["test_predictions"][1]]
            res_dict["test_loss_2"].append(np.mean(test_losses_2))
            res_dict["test_var_2"].append(np.var(test_losses_2))
            K_var = np.asarray(g["K_var"])
            res_dict["kernel_variance"].append(np.sum(K_var))
            res_dict["phase_variance"].append(g["kernel_variance"][()])

            # collect sweep results
            g = data["sweep"]
            res_dict["trial"].append(g["trial"][()])
            res_dict["delta"].append(np.round(g["Delta"][()], decimals=2))

# load data from oscillatory regime
path2 = "results/oscillatory"
for f in os.listdir(path2):

    if "snn_oscillatory_" in f:

        # load data set
        data = h5py.File(f"{path2}/{f}", "r")

        # collect simulation data
        g = data[f"data"]
        res_dict["eta"].append(55.0)
        res_dict["dim"].append(g["dimensionality"][()])
        test_losses_1 = [mse(np.asarray(g["targets"][0]), np.asarray(sig)) for sig in g["test_predictions"][0]]
        res_dict["test_loss_1"].append(np.mean(test_losses_1))
        res_dict["test_var_1"].append(np.var(test_losses_1))
        test_losses_2 = [mse(np.asarray(g["targets"][1]), np.asarray(sig)) for sig in g["test_predictions"][1]]
        res_dict["test_loss_2"].append(np.mean(test_losses_2))
        res_dict["test_var_2"].append(np.var(test_losses_2))
        K_var = np.asarray(g["K_var"])
        res_dict["kernel_variance"].append(np.sum(K_var))
        res_dict["phase_variance"].append(g["kernel_variance"][()])

        # collect sweep results
        g = data["sweep"]
        res_dict["trial"].append(g["trial"][()])
        res_dict["delta"].append(np.round(g["Delta"][()], decimals=2))

# turn dictionary into dataframe
res_df = pd.DataFrame.from_dict(res_dict)

# load pyauto data
auto_dir = "~/PycharmProjects/auto-07p"
a_rs = ODESystem.from_file(f"results/rs.pkl", auto_dir=auto_dir)

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
cmap = sb.crayon_palette(["Midnight Blue", "Indigo", "Wild Blue Yonder"])
etas = [45.0, 55.0, 70.0]
palette = {eta: c for eta, c in zip(etas, cmap)}
err_style = "bars"
err_type = "sd"

# create figure layout
fig = plt.figure(figsize=(6, 4), constrained_layout=True)
grid = fig.add_gridspec(2, 2)

# test loss
ax = fig.add_subplot(grid[0, 1])
# res_df.loc[:, "test_loss_1"] /= np.max(res_df.loc[:, "test_loss_1"])
# res_df.loc[:, "test_loss_2"] /= np.max(res_df.loc[:, "test_loss_2"])
# sb.lineplot(res_df, x="delta", y="test_loss_1", hue="eta", ax=ax2, palette=palette, linestyle="dotted",
#             errorbar=err_type, err_style=err_style)
sb.lineplot(res_df, x="delta", y="test_loss_2", hue="eta", ax=ax, palette=palette, linestyle="solid", legend=False,
            errorbar=err_type, err_style=err_style)
ax.set_xlabel(r"$\Delta_{rs}$")
ax.set_ylabel(r"MSE")
ax.set_title("(B) Averaged MSE across test trials")

# dimensionality
ax = fig.add_subplot(grid[1, 0])
sb.lineplot(res_df, x="delta", y="dim", hue="eta", ax=ax, palette=palette, errorbar=err_type, err_style=err_style)
ax.set_xlabel(r"$\Delta_{rs}$")
ax.set_ylabel(r"$d$")
ax.set_title("(C) Dimensionality")
colors = [l.get_color() for l in ax.get_lines()]

# kernel variance
ax = fig.add_subplot(grid[1, 1])
for eta in etas:
    idx = np.abs(res_df.loc[:, "eta"].values - eta) < 1e-3
    res_df.loc[idx, "phase_variance"] /= np.max(res_df.loc[idx, "phase_variance"])
sb.lineplot(res_df, x="delta", y="phase_variance", hue="eta", ax=ax, palette=palette, errorbar=err_type,
            err_style=err_style)
ax.set_xlabel(r"$\Delta_{rs}$")
ax.set_ylabel(r"$q$")
ax.set_title("(D) Response variance over trials")

# plot bifurcation diagram
ax = fig.add_subplot(grid[0, 0])
a_rs.plot_continuation('PAR(8)', 'PAR(5)', cont=f'D/I:hb1', ax=ax, line_color_stable='#148F77',
                       line_color_unstable='#148F77', line_style_unstable='solid')
a_rs.plot_continuation('PAR(8)', 'PAR(5)', cont=f'D/I:lp3', ax=ax, line_color_stable='#5D6D7E',
                       line_color_unstable='#5D6D7E', line_style_unstable='solid')
a_rs.plot_continuation('PAR(8)', 'PAR(5)', cont=f'D/I:lp4', ax=ax, line_color_stable='#5D6D7E',
                       line_color_unstable='#5D6D7E', line_style_unstable='solid')
ax.set_ylabel(r'$\Delta_{rs}$ (mv)')
ax.set_xlabel(r'$I_{rs}$ (pA)')
ax.set_title(r'(A) RS: $\kappa_{rs} = 100.0$ pA')
ax.set_ylim([0.0, 1.7])
ax.set_xlim([30.0, 80.0])
ax.axvline(x=45.0, color=colors[0], alpha=0.5, linestyle='--', linewidth=0.5, label="node, low FR")
ax.axvline(x=55.0, color=colors[1], alpha=0.5, linestyle='--', linewidth=0.5, label="limit cycle")
ax.axvline(x=70.0, color=colors[2], alpha=0.5, linestyle='--', linewidth=0.5, label="node, high FR")
ax.legend()

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/snn_funcgen.svg')
plt.show()
