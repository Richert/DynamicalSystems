import pickle
import matplotlib.pyplot as plt
plt.switch_backend('tkagg')
from seaborn import scatterplot, lineplot
import os
import sys
import numpy as np
from pandas import DataFrame

# figure settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 12.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0
markersize = 15.0
rowsize = 3
colsize= 3
cmap = "ch:"

# condition
iv = str(sys.argv[-1])
condition = str(sys.argv[-2])
path = str(sys.argv[-3])

# load data
results = {"rep": [], "g": [], "Delta": [], iv: [], "dim_ss": [], "s_mean": [], "s_std": [], "s_norm": [],
           "dim_ir": [], "tau_ir": [], "offset_ir": [], "amp_ir": [], "tau_mf": []}
for file in os.listdir(path):
    if file[:len(condition)] == condition:

        # load data
        data = pickle.load(open(f"{path}/{file}", "rb"))

        # condition information
        f = file.split("_")
        rep = int(f[-1].split(".")[0])
        results["rep"].append(rep)
        results["g"].append(data["g"])
        results["Delta"].append(data["Delta_e"])
        results[iv].append(data[iv])
        # results[iv].append(float(f[-2][1:]))

        # steady-state analysis
        results["dim_ss"].append(data["dim_ss"])
        results["s_mean"].append(np.mean(data["s_mean"])*1e3)
        results["s_std"].append(np.mean(data["s_std"]))
        results["s_norm"].append(results["s_std"][-1]*1e3/results["s_mean"][-1])

        # impulse response analysis
        results["dim_ir"].append(data["dim_ir"])
        results["tau_ir"].append(data["params_ir"][-2])
        results["offset_ir"].append(data["params_ir"][0])
        results["amp_ir"].append(data["params_ir"][2])
        results["tau_mf"].append(data["mf_params_ir"][-1])

# create dataframe
df = DataFrame.from_dict(results)

# # filter results
# min_iv = 0.0
# max_iv = 15.0
# df = df.loc[df[iv] >= min_iv, :]
# df = df.loc[df[iv] <= max_iv, :]

# plotting line plots for steady state regime
ivs = np.unique(df.loc[:, iv].values)
fig = plt.figure(figsize=(12, 3*len(ivs)))
grid = fig.add_gridspec(nrows=len(ivs), ncols=3)
for i, p in enumerate(ivs):
    df_tmp = df.loc[df[iv] == p, :]
    for j, y in enumerate(["dim_ss", "s_mean", "s_norm"]):
        ax = fig.add_subplot(grid[i, j])
        lineplot(df_tmp, x="g", hue="Delta", y=y, ax=ax, palette=cmap)
        if j == 1:
            ax.set_title(f"{iv} = {np.round(p, decimals=2)}")
fig.suptitle("Steady-Sate Dynamics")
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
fig.canvas.draw()
plt.savefig(f'{path}/figures/steady_state_dynamics.pdf')

# plotting line plots for impulse response
fig = plt.figure(figsize=(12, 3*len(ivs)))
grid = fig.add_gridspec(nrows=len(ivs), ncols=3)
for i, p in enumerate(ivs):
    df_tmp = df.loc[df[iv] == p, :]
    for j, y in enumerate(["dim_ir", "tau_ir", "amp_ir"]):
        ax = fig.add_subplot(grid[i, j])
        lineplot(df_tmp, x="g", hue="Delta", y=y, ax=ax, palette=cmap)
        if j == 1:
            ax.set_title(f"{iv} = {np.round(p, decimals=2)}")
fig.suptitle("Impulse Response")
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
fig.canvas.draw()
plt.savefig(f'{path}/figures/impulse_response.pdf')

# plotting scatter plots for dimensionality
fig = plt.figure(figsize=(2*rowsize, 2*colsize))
grid = fig.add_gridspec(ncols=2, nrows=2)
ax = fig.add_subplot(grid[0, 0])
scatterplot(df, x="s_norm", y="dim_ss", hue="Delta", style=iv, palette=cmap, legend=True, ax=ax, s=markersize)
ax.set_xlabel(r"$\mathrm{std}(r) / \mathrm{mean}(r)$")
ax.set_ylabel(r"$D$")
ax = fig.add_subplot(grid[0, 1])
scatterplot(df, x="s_norm", y="dim_ss", hue="g", style=iv, palette=cmap, legend=True, ax=ax, s=markersize)
ax.set_xlabel(r"$\mathrm{std}(r) / \mathrm{mean}(r)$")
ax.set_ylabel(r"")
ax = fig.add_subplot(grid[1, 0])
scatterplot(df, x="dim_ir", y="dim_ss", hue="Delta", style=iv, palette=cmap, legend=True, ax=ax, s=markersize)
ax.set_xlabel(r"$D_{ir}$")
ax.set_ylabel(r"$D_{ss}$")
ax = fig.add_subplot(grid[1, 1])
scatterplot(df, x="dim_ir", y="dim_ss", hue="g", style=iv, palette=cmap, legend=True, ax=ax, s=markersize)
ax.set_xlabel(r"$D_{ir}$")
ax.set_ylabel(r"")
fig.suptitle("Dimensionality: Steady-State vs. Impulse Response")
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
fig.canvas.draw()
plt.savefig(f'{path}/figures/dimensionality_ir_ss.pdf')

# filter results
min_alpha = 0.2
df = df.loc[df["amp_ir"] >= min_alpha, :]

# plotting scatter plots for impulse response time constants
fig = plt.figure(figsize=(2*rowsize, 2*colsize))
grid = fig.add_gridspec(ncols=2, nrows=2)
ax = fig.add_subplot(grid[0, 0])
scatterplot(df, x="dim_ir", y="tau_ir", hue="Delta", style=iv, palette=cmap, legend=True, ax=ax)
ax.set_xlabel(r"$D_{ir}$")
ax.set_ylabel(r"$\tau$")
ax = fig.add_subplot(grid[0, 1])
scatterplot(df, x="dim_ir", y="tau_ir", hue="g", style=iv, palette=cmap, legend=True, ax=ax)
ax.set_xlabel(r"$D_{ir}$")
ax.set_ylabel(r"$\tau$")
ax = fig.add_subplot(grid[1, 0])
scatterplot(df, x="dim_ss", y="tau_ir", hue="Delta", style=iv, palette=cmap, legend=True, ax=ax)
ax.set_xlabel(r"$D_{ss}$")
ax.set_ylabel(r"$\tau$")
ax = fig.add_subplot(grid[1, 1])
scatterplot(df, x="dim_ss", y="tau_ir", hue="g", style=iv, palette=cmap, legend=True, ax=ax)
ax.set_xlabel(r"$D_{ss}$")
ax.set_ylabel(r"$\tau$")
fig.suptitle("Dimensionality vs. Impulse Response Time Constant")
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
fig.canvas.draw()
plt.savefig(f'{path}/figures/dimensionality_tau.pdf')

plt.show()
