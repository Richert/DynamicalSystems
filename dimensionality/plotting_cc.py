import pickle
from seaborn import heatmap, scatterplot, lineplot
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from pandas import DataFrame
from custom_functions import impulse_response_fit, alpha

# condition
condition = str(sys.argv[-1])
path = str(sys.argv[-2])

# load data
results = {"rep": [], "g": [], "Delta": [], "p": [], "dim_ss": [], "s_mean": [], "s_std": [], "s_norm": [],
           "dim_ir": [], "tau_ir": [], "offset_ir": [], "amp_ir": []}
for file in os.listdir(path):
    if file[:len(condition)] == condition:

        # load data
        data = pickle.load(open(f"{path}/{file}", "rb"))

        # condition information
        f = file.split("_")
        rep = int(f[-1].split(".")[0])
        p = float(f[-2][1:])/10.0
        results["rep"].append(rep)
        results["g"].append(data["g"])
        results["Delta"].append(data["Delta"])
        results["p"].append(p)

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

# create dataframe
df = DataFrame.from_dict(results)

# filter results
# min_g = 0.0
# max_g = 25.0
# df = df.loc[df["g"] >= min_g, :]
# df = df.loc[df["g"] <= max_g, :]

# plotting line plots
fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
for ax, y, title in zip(axes, ["dim_ss", "s_mean", "s_norm"], ["Participation Ratio", "mean(fr)", "std(fr)/mean(fr)"]):
    lineplot(df, x="g", hue="Delta", y=y, ax=ax)
    ax.set_title(title)
plt.tight_layout()

# plotting scatter plots
fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(12, 4))
ax = axes[0, 0]
scatterplot(df, x="s_norm", y="dim_ss", hue="Delta", palette="tab10", legend=True, ax=ax)
ax = axes[1, 0]
scatterplot(df, x="s_norm", y="dim_ss", hue="g", palette="tab10", legend=True, ax=ax)
ax = axes[0, 1]
scatterplot(df, x="dim_ir", y="dim_ss", hue="Delta", palette="tab10", legend=True, ax=ax)
ax = axes[1, 1]
scatterplot(df, x="dim_ir", y="dim_ss", hue="g", palette="tab10", legend=True, ax=ax)
ax = axes[0, 1]
scatterplot(df, x="dim_ir", y="tau_ir", hue="Delta", palette="tab10", legend=True, ax=ax)
ax = axes[1, 1]
scatterplot(df, x="dim_ir", y="tau_ir", hue="g", palette="tab10", legend=True, ax=ax)
plt.tight_layout()

plt.show()
