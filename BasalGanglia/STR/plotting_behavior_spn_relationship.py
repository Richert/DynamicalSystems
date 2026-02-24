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
path = f"/mnt/kennedy_lab_data/Parkerlab/neural_data"
results = read_csv(f"{path}/spn_dimensionality_data.csv")
connectivity = pickle.load(open(f"{path}/spn_connectivity_data.pkl", "rb"))

# choose plotting conditions
drugs = ["clozapine", "olanzapine", "xanomeline", "MP10", "haloperidol", "M4PAM", "SCH23390",
         "SCH39166", "SEP363856", "SKF38393"]
variables = ["p(v)", "mean rate", "rate fluctuation", "dimensionality"]

# plotting
sb.set_palette("colorblind")
results.sort_values(["condition", "dose"], inplace=True)
for key in ["fano factor"]:
    for drug in drugs:
        if d12_combined:
            fig, ax = plt.subplots(figsize=(10, 6))
            sb.lineplot(df.loc[df.loc[:, "drug"] == drug, :],
                        x="v", y=key, hue="dose", style="condition", ax=ax,
                        markers=True, dashes=True, err_style='bars', errorbar="se")
            ax.set_title(drug)
            plt.tight_layout()
            fig.canvas.draw()
            fig.savefig(f"/home/rgast/data/parker_data/{drug}_{key}_d1_d2_combined.svg")
        else:
            fig, axes = plt.subplots(ncols=2, figsize=(12, 6), sharex=True, sharey=True)
            fig.suptitle(drug)
            for i, d in enumerate(["D1", "D2"]):
                ax = axes[i]
                sb.lineplot(df.loc[(df.loc[:, "drug"] == drug) & (df.loc[:, "neuron_type"] == d), :],
                            x="v", y=key, hue="dose", style="condition", ax=ax,
                            markers=True, dashes=True, err_style='bars', errorbar="se")
                ax.set_title(f"{d}")
            plt.tight_layout()
            fig.canvas.draw()
            fig.savefig(f"/home/rgast/data/parker_data/{drug}_{key}_d1_d2_split.svg")
plt.show()
