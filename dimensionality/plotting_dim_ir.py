import pickle
from seaborn import heatmap, scatterplot, lineplot
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from pandas import DataFrame

# condition
condition = str(sys.argv[-1])
path = str(sys.argv[-2])

# load data
results = {"rep": [], "g": [], "Delta": [], "dim_ss": [], "dim_ir": [], "s_mean": [], "s_std": [], "s_norm": [],
           "ir_tau": []}
for file in os.listdir(path):
    if file[:len(condition)] == condition:
        data = pickle.load(open(f"{path}/{file}", "rb"))
        rep = file.split("_")[-1]
        rep = int(rep.split(".")[0])
        results["rep"].append(rep)
        results["g"].append(data["g"])
        results["Delta"].append(data["Delta"])
        results["dim_ss"].append(data["dim"])
        results["dim_ir"].append(data["dim"])
        results["s_mean"].append(np.mean(data["s_mean"]))
        results["s_std"].append(np.mean(data["s_std"]))
        results["s_norm"].append(results["s_std"][-1]/results["s_mean"][-1])
        results["ir_tau"].append(data["ir_params"][2])

# create dataframe
df = DataFrame.from_dict(results)

# filter results
min_g = 0.0
max_g = 25.0
df = df.loc[df["g"] >= min_g, :]
df = df.loc[df["g"] <= max_g, :]

# reshape results into 2D tables
dim_ss = df.pivot_table(values="dim_ss", index="g", columns="Delta")
dim_ir = df.pivot_table(values="dim_ir", index="g", columns="Delta")
fr_std = df.pivot_table(values="s_std", index="g", columns="Delta")
fr_norm = df.pivot_table(values="s_norm", index="g", columns="Delta")
ir_tau = df.pivot_table(values="ir_tau", index="g", columns="Delta")

# plotting 2D plots
for data, title in zip([dim_ss, dim_ir, fr_std, fr_norm, ir_tau],
                       ["dim(ss)", "dim(ir)", "std(fr)", "std(fr)/mean(fr)", "tau(ir)"]):
    _, ax = plt.subplots(figsize=(5, 5))
    im = heatmap(data, ax=ax)
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Delta")
    ax.set_ylabel("g")
    ax.set_title(title)
    plt.tight_layout()

# plotting 1D line plots
for y, title in zip(["dim_ss", "dim_ir", "ir_tau"], ["dim(ss)", "dim(ir)", "tau(ir)"]):
    _, ax = plt.subplots(figsize=(8, 4))
    scatterplot(df, x="Delta", y=y, hue="g", palette="tab10", legend=False, ax=ax)
    lineplot(df, x="Delta", y=y, hue="g", palette="tab10", ax=ax)
    ax.set_title(title)
    plt.tight_layout()

# plotting 1D scatter plots
for y, title in zip(["dim_ss", "dim_ir", "ir_tau"], ["dim(ss)", "dim(ir)", "tau(ir)"]):
    fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
    ax = axes[0]
    scatterplot(df, x="s_norm", y=y, hue="Delta", style="g", ax=ax)
    ax.set_title("std(fr)/mean(fr)")
    ax = axes[1]
    scatterplot(df, x="s_std", y=y, hue="Delta", style="g", ax=ax)
    ax.set_title("std(fr)")
    fig.suptitle(title)
    plt.tight_layout()

plt.show()
