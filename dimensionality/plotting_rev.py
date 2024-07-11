import pickle
from seaborn import heatmap, scatterplot, lineplot
import matplotlib.pyplot as plt
import os
import numpy as np
from pandas import DataFrame

# condition
condition = "rev_"

# load data
path = "/media/fsmresfiles/richard_data/numerics/dimensionality"
results = {"rep": [], "g": [], "Delta": [], "dim": [], "s_mean": [], "s_std": [], "s_norm": []}
for file in os.listdir(path):
    if file[:len(condition)] == condition:
        f_tmp = file.split("_")
        data = pickle.load(open(f"{path}/{file}", "rb"))
        results["rep"].append(int(f_tmp[-1][:-2]))
        results["g"].append(data["g"])
        results["Delta"].append(data["Delta"])
        results["E_r"].append(-float(f_tmp[4][1:]))
        results["dim"].append(data["dim"])
        results["s_mean"].append(np.mean(data["s_mean"]))
        results["s_std"].append(np.mean(data["s_std"]))
        results["s_norm"].append(results["s_std"][-1]/results["s_mean"][-1])

# create dataframe
df = DataFrame.from_dict(results)

# plotting
E_rs = np.unique(df.loc[:, "E_r"].values)
for reversal_potential in E_rs:

    df = df.loc[(df["E_r"] - reversal_potential) < 1e-3, :]

    fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
    ax = axes[0]
    scatterplot(df, x="Delta", y="dim", hue="g", palette="tab10", legend=False, ax=ax)
    lineplot(df, x="Delta", y="dim", hue="g", palette="tab10", ax=ax)
    ax = axes[1]
    scatterplot(df, x="s_norm", y="dim", hue="Delta", style="g", ax=ax)
    ax = axes[2]
    scatterplot(df, x="s_std", y="dim", hue="Delta", style="g", ax=ax)
    fig.suptitle(f"E_r = {reversal_potential}")
    plt.tight_layout()

plt.show()
