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
colsize= 4
cmap = "ch:"

# condition
condition = str(sys.argv[-1])
path = str(sys.argv[-2])

# load data
results = {"rep": [], "g": [], "Delta": [], "p": [], "dim_ss": [], "s_mean": [], "s_std": [], "s_norm": [],
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
        results["Delta"].append(data["Delta"])
        results["p"].append(data["p"])

        # steady-state analysis
        results["dim_ss"].append(data["dim_ss"])
        results["s_mean"].append(np.mean(data["s_mean"]))
        results["s_std"].append(np.mean(data["s_std"]))
        results["s_norm"].append(results["s_std"][-1]/results["s_mean"][-1])

        # impulse response analysis
        results["dim_ir"].append(data["dim_ir"])
        results["tau_ir"].append(data["params_ir"][-1])
        results["offset_ir"].append(data["params_ir"][0])
        results["amp_ir"].append(data["params_ir"][2])
        results["tau_mf"].append(data["mf_params_ir"][-1])

# create dataframe
df = DataFrame.from_dict(results)

# filter results
# min_g = 0.0
# max_g = 25.0
# df = df.loc[df["g"] >= min_g, :]
# df = df.loc[df["g"] <= max_g, :]

# plotting line plots for steady state regime
ps = np.unique(df.loc[:, "p"].values)
fig = plt.figure(figsize=(12, 3*len(ps)))
grid = fig.add_gridspec(nrows=len(ps), ncols=3)
for i, p in enumerate(ps):
    df_tmp = df.loc[df["p"] == p, :]
    for j, y in enumerate(["dim_ss", "s_mean", "s_norm"]):
        ax = fig.add_subplot(grid[i, j])
        lineplot(df_tmp, x="g", hue="Delta", y=y, ax=ax, palette=cmap)
        if j == 1:
            ax.set_title(f"p = {np.round(p, decimals=2)}")
fig.suptitle("Steady-Sate Dynamics")
plt.tight_layout()

# plotting line plots for impulse response
fig = plt.figure(figsize=(12, 3*len(ps)))
grid = fig.add_gridspec(nrows=len(ps), ncols=3)
for i, p in enumerate(ps):
    df_tmp = df.loc[df["p"] == p, :]
    for j, y in enumerate(["dim_ir", "tau_ir", "amp_ir"]):
        ax = fig.add_subplot(grid[i, j])
        lineplot(df_tmp, x="g", hue="Delta", y=y, ax=ax, palette=cmap)
        if j == 1:
            ax.set_title(f"p = {np.round(p, decimals=2)}")
fig.suptitle("Impulse Response")
plt.tight_layout()

# plotting scatter plots for dimensionality
fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(12, 6))
ax = axes[0, 0]
scatterplot(df, x="s_norm", y="dim_ss", hue="Delta", style="p", palette=cmap, legend=True, ax=ax)
ax = axes[1, 0]
scatterplot(df, x="s_norm", y="dim_ss", hue="g", style="p", palette=cmap, legend=True, ax=ax)
ax = axes[0, 1]
scatterplot(df, x="dim_ir", y="dim_ss", hue="Delta", style="p", palette=cmap, legend=True, ax=ax)
ax = axes[1, 1]
scatterplot(df, x="dim_ir", y="dim_ss", hue="g", style="p", palette=cmap, legend=True, ax=ax)
fig.suptitle("Control of Dimensionality")
plt.tight_layout()

# plotting scatter plots for impulse response time constants
fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(12, 6))
ax = axes[0, 0]
scatterplot(df, x="dim_ir", y="tau_ir", hue="Delta", style="p", palette=cmap, legend=True, ax=ax)
ax = axes[1, 0]
scatterplot(df, x="dim_ir", y="tau_ir", hue="g", style="p", palette=cmap, legend=True, ax=ax)
ax = axes[0, 1]
scatterplot(df, x="dim_ir", y="tau_mf", hue="Delta", style="p", palette=cmap, legend=True, ax=ax)
ax = axes[1, 1]
scatterplot(df, x="dim_ir", y="tau_mf", hue="g", style="p", palette=cmap, legend=True, ax=ax)
fig.suptitle("Impulse Response Time Constant")
plt.tight_layout()

plt.show()
