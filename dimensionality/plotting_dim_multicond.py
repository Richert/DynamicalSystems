import pickle
from seaborn import heatmap, scatterplot, lineplot
import matplotlib.pyplot as plt
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
colsize= 4
cmap = "ch:"

titles = {"exc": "glutamatergic", "inh": "gabaergic", "spn": "SPNs, e/i", "inh2": "gabaergic, e/i",
          "spn2": "SPNs, e/i, fixed W"}

# condition
path = str(sys.argv[1])
condition = [str(sys.argv[2+i]) for i in range(len(sys.argv[2:]))]

# load data
results = {"rep": [], "g": [], "Delta": [], "dim_ss": [], "dim_ir": [], "s_mean": [], "s_std": [], "s_norm": [],
           "ir_tau": [], "cond": []}
for file in os.listdir(path):
    c = file.split("_")[0]
    if c in condition:
        data = pickle.load(open(f"{path}/{file}", "rb"))
        rep = file.split("_")[-1]
        rep = int(rep.split(".")[0])
        results["cond"].append(c)
        results["rep"].append(rep)
        results["g"].append(data["g"])
        results["Delta"].append(data["Delta"])
        results["dim_ss"].append(data["dim_ss"])
        results["dim_ir"].append(data["dim_ir"])
        results["s_mean"].append(np.mean(data["s_mean"]))
        results["s_std"].append(np.mean(data["s_std"]))
        results["s_norm"].append(results["s_std"][-1]/results["s_mean"][-1])
        results["ir_tau"].append(data["ir_params"][2])

# create dataframe
df = DataFrame.from_dict(results)

# filter results
min_g = 0.0
max_g = 30.0
df = df.loc[df["g"] >= min_g, :]
df = df.loc[df["g"] <= max_g, :]

# plotting 1D scatter plot for firing rate heterogeneity vs dimensionality in the steady state
fig = plt.figure(1, figsize=(colsize*2, len(condition)*rowsize))
grid = fig.add_gridspec(nrows=len(condition), ncols=2)
for i, c in enumerate(condition):
    df_tmp = df.loc[df["cond"] == c, :]
    ax = fig.add_subplot(grid[i, 0])
    scatterplot(df_tmp, x="s_norm", y="dim_ss", hue="Delta", ax=ax, s=markersize, palette=cmap)
    ax.set_xlabel("")
    if i == 0:
        ax.set_ylabel(r"$D(r)$")
    else:
        ax.set_ylabel("")
    if i == len(condition)-1:
        ax.set_xlabel(r"$\mathrm{std}(r) / \mathrm{mean}(r)$")
    else:
        ax.set_xlabel("")
    ax.set_title(titles[c.split("_")[0]])
    ax = fig.add_subplot(grid[i, 1])
    scatterplot(df_tmp, x="s_norm", y="dim_ss", hue="g", ax=ax, s=markersize, palette=cmap)
    ax.set_ylabel("")
    if i == len(condition) - 1:
        ax.set_xlabel(r"$\mathrm{std}(r) / \mathrm{mean}(r)$")
    else:
        ax.set_xlabel("")
fig.suptitle("Steady-State Network Dimensionality")
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
fig.canvas.draw()
plt.savefig(f'{path}/figures/scatter_ss.pdf')

# plotting 1D scatter plot for firing rate heterogeneity vs dimensionality during the impulse response
fig = plt.figure(2, figsize=(colsize*2, len(condition)*rowsize))
grid = fig.add_gridspec(nrows=len(condition), ncols=2)
for i, c in enumerate(condition):
    df_tmp = df.loc[df["cond"] == c, :]
    ax = fig.add_subplot(grid[i, 0])
    scatterplot(df_tmp, x="s_norm", y="dim_ir", hue="Delta", ax=ax, s=markersize, palette=cmap)
    ax.set_xlabel("")
    if i == 0:
        ax.set_ylabel(r"$D(r)$")
    else:
        ax.set_ylabel("")
    if i == len(condition)-1:
        ax.set_xlabel(r"$\mathrm{std}(r) / \mathrm{mean}(r)$")
    else:
        ax.set_xlabel("")
    ax.set_title(titles[c.split("_")[0]])
    ax = fig.add_subplot(grid[i, 1])
    scatterplot(df_tmp, x="s_norm", y="dim_ir", hue="g", ax=ax, s=markersize, palette=cmap)
    ax.set_ylabel("")
    if i == len(condition) - 1:
        ax.set_xlabel(r"$\mathrm{std}(r) / \mathrm{mean}(r)$")
    else:
        ax.set_xlabel("")
fig.suptitle("Impulse Response Network Dimensionality")
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
fig.canvas.draw()
plt.savefig(f'{path}/figures/scatter_ir.pdf')

# plotting 1D line plot
fig = plt.figure(3, figsize=(colsize, len(condition)*rowsize))
grid = fig.add_gridspec(nrows=len(condition), ncols=1)
for i, c in enumerate(condition):
    df_tmp = df.loc[df["cond"] == c, :]
    ax = fig.add_subplot(grid[i, 0])
    scatterplot(df_tmp, x="Delta", y="dim_ss", hue="g", palette=cmap, legend=False, ax=ax, s=markersize)
    lineplot(df_tmp, x="Delta", y="dim_ss", hue="g", palette=cmap, ax=ax)
    ax.set_ylabel(r"$D(r)$")
    if i == len(condition) - 1:
        ax.set_xlabel(r"$\Delta$")
    else:
        ax.set_xlabel("")
    ax.set_title(titles[c.split("_")[0]])
fig.suptitle("ABC")
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
fig.canvas.draw()
plt.savefig(f'{path}/figures/line_ss.pdf')

plt.show()
