import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["axes.labelsize"] = 14
matplotlib.rcParams["lines.markersize"] = 12.0
matplotlib.rcParams["lines.linewidth"] = 2.0
import seaborn as sb
from pandas import DataFrame, read_csv
import pickle

# load data
path = f"/home/rgast/data/parker_data"
df = read_csv(f"{path}/spn_behavior_v_single_window.csv")
df.sort_values(["condition", "neuron_type"], inplace=True)

# choose plotting conditions
drugs = [
    "MP10", "haloperidol"
    # "olanzapine", "clozapine", "xanomeline", "M4PAM", "SCH23390", "SCH39166", "SEP363856", "SKF38393"
]
behaviors = np.unique(df.loc[:, "behavior"])

# plotting
sb.set_palette("colorblind")
for key in ["D(C)", "D(r)", "mean(r)", "p(behavior)"]:
    for drug in drugs:
        fig, axes = plt.subplots(ncols=2, figsize=(12, 5))
        for i, n in enumerate(["D1", "D2"]):
            ax = axes[i]
            sb.lineplot(df.loc[(df.loc[:, "drug"] == drug) & (df.loc[:, "neuron_type"] == n), :],
                        x="behavior", y=key, hue="condition", ax=ax,
                        markers=True, dashes=True, err_style='bars', errorbar="se")
        fig.suptitle(f"Drug = {drug}")
        plt.tight_layout()
        fig.canvas.draw()
plt.show()
