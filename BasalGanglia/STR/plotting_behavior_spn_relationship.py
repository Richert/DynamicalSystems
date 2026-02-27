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
results.sort_values(["condition", "dose"], inplace=True)
raw_data = pickle.load(open(f"{path}/spn_dimensionality_data.pkl", "rb"))
drug_vals = raw_data["drugs"]
cond_vals = raw_data["conditions"]
dose_vals = raw_data["dose"]
mice = raw_data["mouse"]
velocity = np.asarray(raw_data["v(t)"])
pc1 = np.asarray(raw_data["pc1"])
Cs = np.asarray(raw_data["C"])

# choose plotting conditions
drugs = ["clozapine", "olanzapine", "xanomeline", "MP10", "haloperidol", "M4PAM", "SCH23390",
         "SCH39166", "SEP363856", "SKF38393"]
variables = ["p(v)", "mean rate", "rate fluctuation", "dimensionality"]
d12_combined = True

# plotting
sb.set_palette("colorblind")
for drug in drugs:

    # reduce data to drug
    idx_drug = drug_vals == drug
    df = results.loc[results.loc[:, "drug"] == drug, :]

    if d12_combined:

        fig = plt.figure(figsize=(12, 10))
        grid = fig.add_gridspec(nrows=6, ncols=4)

        # plot animal velocity and spn dynamics
        for i, (c, d) in enumerate([("veh", "Vehicle"), ("amph", "Vehicle"), ("amph", "HighDose")]):
            ax = fig.add_subplot(grid[i*2:(i+1)*2, :2])
            idx_cond = cond_vals == c
            idx_dose = dose_vals == d
            m = mice[idx_drug & idx_cond & idx_dose]
            mouse = np.random.randint(len(m))
            v = velocity[idx_drug & idx_cond & idx_dose][mouse]
            r = pc1[idx_drug & idx_cond & idx_dose][mouse]
            ax.plot(v, label="velocity")
            ax2 = ax.twinx()
            ax2.plot(r, label="PC1")
            ax.legend()
            ax.set_title(f"Condition: {c}, Dose: {d}, Mouse: {m[mouse]}")

        # plot neural covariance matrices 
        sb.lineplot(,
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
