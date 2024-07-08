import pickle
from seaborn import heatmap, scatterplot, lineplot
import matplotlib.pyplot as plt
import os
import numpy as np
from pandas import DataFrame


# load data
path = "/home/rgf3807/PycharmProjects/DynamicalSystems/Izhikevich/results/snn_dim"
results = {"rep": [], "g": [], "Delta": [], "dim": [], "s_mean": [], "s_std": [], "s_norm": []}
for file in os.listdir(path):
    if file[:3] == "ir_":
        data = pickle.load(open(f"{path}/{file}", "rb"))
        results["rep"].append(int(file.split("_")[-1][:-2]))
        results["g"].append(data["g"])
        results["Delta"].append(data["Delta"])
        results["dim"].append(data["dim"])
        results["s_mean"].append(np.mean(data["s_mean"]))
        results["s_std"].append(np.mean(data["s_std"]))
        results["s_norm"].append(results["s_std"][-1]/results["s_mean"][-1])

# create dataframe
df = DataFrame.from_dict(results)

# filter results
min_g = 0.0
max_g = 25.0
df = df.loc[df["g"] >= min_g, :]
df = df.loc[df["g"] <= max_g, :]

# reshape results into 2D tables
dim = df.pivot_table(values="dim", index="g", columns="Delta")
fr_std = df.pivot_table(values="s_std", index="g", columns="Delta")
fr_norm = df.pivot_table(values="s_norm", index="g", columns="Delta")

# plotting 2D plots
fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
for ax, data, title in zip(axes, [dim, fr_std, fr_norm], ["Participation Ratio", "std(fr)", "std(fr)/mean(fr)"]
                           ):
    heatmap(data, ax=ax)
    ax.set_title(title)
plt.tight_layout()

# plotting 1D plots
fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
ax = axes[0]
scatterplot(df, x="Delta", y="dim", hue="g", palette="tab10", legend=False, ax=ax)
lineplot(df, x="Delta", y="dim", hue="g", palette="tab10", ax=ax)
ax = axes[1]
scatterplot(df, x="s_norm", y="dim", hue="Delta", style="g", ax=ax)
ax = axes[2]
scatterplot(df, x="s_std", y="dim", hue="Delta", style="g", ax=ax)
plt.tight_layout()
plt.show()
