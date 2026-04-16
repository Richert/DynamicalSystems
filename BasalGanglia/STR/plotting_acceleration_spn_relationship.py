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
results = read_csv(f"{path}/spn_dimensionality_acceleration.csv")
results.sort_values(["condition", "dose"], inplace=True)

# choose plotting conditions
drugs = [
    "olanzapine", "clozapine"
    # "MP10", "xanomeline", "haloperidol", "xanomeline", "M4PAM", "SCH23390", "SCH39166", "SEP363856", "SKF38393"
]
d12_combined = False

# plotting
sb.set_palette("colorblind")
for key in ["D(C)", "D_b(C)", "D(r)", "mean(r)", "p(a)", "v"]:
    for drug in drugs:
        if d12_combined:
            fig, ax = plt.subplots(figsize=(10, 6))
            sb.lineplot(results.loc[results.loc[:, "drug"] == drug, :],
                        x="a", y=key, hue="dose", style="condition", ax=ax,
                        markers=True, dashes=True, err_style='bars', errorbar="se")
            ax.set_title(drug)
            plt.tight_layout()
            fig.canvas.draw()
        else:
            fig, axes = plt.subplots(ncols=2, figsize=(12, 6), sharex=True, sharey=True)
            fig.suptitle(drug)
            for i, d in enumerate(["D1", "D2"]):
                ax = axes[i]
                sb.lineplot(results.loc[(results.loc[:, "drug"] == drug) & (results.loc[:, "neuron_type"] == d), :],
                            x="a", y=key, hue="dose", style="condition", ax=ax,
                            markers=True, dashes=True, err_style='bars', errorbar="se")
                ax.set_title(f"{d}")
            plt.tight_layout()
            fig.canvas.draw()
plt.show()
