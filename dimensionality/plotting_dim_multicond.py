import pickle
from seaborn import heatmap, scatterplot, lineplot
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from pandas import DataFrame

# figure settings
print(f"Plotting backend: {plt.rcParams['backend']}")
# plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=False)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 400
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 1.0
markersize = 1.0

# condition
path = str(sys.argv[1])
condition = [str(sys.argv[2+i]) for i in range(len(sys.argv[2:]))]

# load data
results = {"rep": [], "g": [], "Delta": [], "dim_ss": [], "dim_ir": [], "s_mean": [], "s_std": [], "s_norm": [],
           "ir_tau": [], "cond": []}
for file in os.listdir(path):
    if file[:len(condition[0])] in condition:
        data = pickle.load(open(f"{path}/{file}", "rb"))
        rep = file.split("_")[-1]
        rep = int(rep.split(".")[0])
        results["cond"].append(file[:len(condition[0])])
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
fig = plt.figure(figsize=(int(4*len(condition)), 4))
grid = fig.add_gridspec(nrows=2, ncols=len(condition))
for i, c in enumerate(condition):
    df_tmp = df.loc[df["cond"] == c, :]
    ax = fig.add_subplot(grid[0, i])
    scatterplot(df, x="s_norm", y="dim_ss", hue="Delta", ax=ax, markersize=markersize)
    ax.set_title(c.split("_")[0])
    ax = fig.add_subplot(grid[1, i])
    scatterplot(df, x="s_norm", y="dim_ss", hue="g", ax=ax, markersize=markersize)
    ax.set_xlabel("std(r)/mean(r)")
    ax.set_ylabel("dim(ss)")
fig.suptitle("Network Dimensionality in the Steady-State Condition")
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
fig.canvas.draw()
plt.savefig(f'{path}/figures/dimensionality_ss.pdf')

# plotting 1D scatter plot for firing rate heterogeneity vs dimensionality during the impulse response
fig = plt.figure(figsize=(int(4*len(condition)), 4))
grid = fig.add_gridspec(nrows=2, ncols=len(condition))
for i, c in enumerate(condition):
    df_tmp = df.loc[df["cond"] == c, :]
    ax = fig.add_subplot(grid[0, i])
    scatterplot(df, x="s_norm", y="dim_ir", hue="Delta", ax=ax, markersize=markersize)
    ax.set_title(c.split("_")[0])
    ax = fig.add_subplot(grid[1, i])
    scatterplot(df, x="s_norm", y="dim_ir", hue="g", ax=ax, markersize=markersize)
    ax.set_xlabel("std(r)/mean(r)")
    ax.set_ylabel("dim(ir)")
fig.suptitle("Network Dimensionality in the Impulse Response Condition")
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
fig.canvas.draw()
plt.savefig(f'{path}/figures/dimensionality_ir.pdf')

# plotting 1D scatter plot for impulse response decay vs dimensionality during the impulse response
fig = plt.figure(figsize=(int(4*len(condition)), 4))
grid = fig.add_gridspec(nrows=2, ncols=len(condition))
for i, c in enumerate(condition):
    df_tmp = df.loc[df["cond"] == c, :]
    ax = fig.add_subplot(grid[0, i])
    scatterplot(df, x="ir_tau", y="dim_ir", hue="Delta", ax=ax, markersize=markersize)
    ax.set_title(c.split("_")[0])
    ax = fig.add_subplot(grid[1, i])
    scatterplot(df, x="ir_tau", y="dim_ir", hue="g", ax=ax, markersize=markersize)
    ax.set_xlabel("tau(ir)")
    ax.set_ylabel("dim(ir)")
fig.suptitle("Network Dimensionality vs Decay Time Consteant in the Impulse Response Condition")
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
fig.canvas.draw()
plt.savefig(f'{path}/figures/time_constant_ir.pdf')

plt.show()
