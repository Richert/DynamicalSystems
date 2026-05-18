import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pandas import read_csv
import pickle
from scipy.ndimage import gaussian_filter1d

# preparations
##############

# load all-mice 1D data
condition = "p"
path = f"/home/rgast/data/parker_data"
df = read_csv(f"{path}/spn_behavior_{condition}.csv")

# load all-mice multi-D data
multi_data = pickle.load(open(f"{path}/spn_behavior_{condition}.pkl", "rb"))

# load single mouse data
drug = "haloperidol"
dose = "Vehicle"
mouse = "m972"
single_data = pickle.load(open(f"{path}/mouse_data_{mouse}_{drug}_{dose}.pkl", "rb"))

# choose plotting conditions
drugs = [
    "haloperidol", "xanomeline", "MP10"
    # "MP10", "olanzapine", "clozapine", "xanomeline", "M4PAM", "SCH23390", "SCH39166", "SEP363856", "SKF38393"
]
behaviors = ["p0", "p1", "p2", "p3"]

# post-analysis processing
idx_c = df.loc[:, "condition"] != "A + Low Dose"
df = df.loc[idx_c, :]
multi_data = {key: val[idx_c] for key, val in multi_data.items()}
idx_c2 = df.loc[:, "condition"] == "A + High Dose"
df.loc[idx_c2, "condition"] = "A + Drug"
multi_data["condition"][idx_c2] = "A + Drug"
df_sorted = df.sort_values(["condition", "behavior", "SPNs"], inplace=False, ascending=[False, True, True])

# data parameters
rate_bins = np.linspace(0.0, 0.5, num=20)

# plotting
##########

# plot settings
matplotlib.use("TkAgg")
plt.rcParams["font.size"] = 16.0
plt.rcParams["lines.markersize"] = 12.0
plt.rcParams["lines.linewidth"] = 2.0
import seaborn as sb
sb.set_palette("colorblind")

# figure layout
fig = plt.figure(figsize=(14, 16), layout="tight")
grid = fig.add_gridspec(nrows=5, ncols=12)

# first row: velocity autocorrelations
for i, d in enumerate(drugs):

    ax = fig.add_subplot(grid[0, i * 4:(i + 1) * 4])
    idx = multi_data["drug"] == d

    for j, c in enumerate(["Control", "Amphetamine", "A + Drug"]):

        idx2 = multi_data["condition"] == c
        vc = multi_data["AC(v)"][idx & idx2]
        vc = np.mean(vc, axis=0)
        time_lags = (np.arange(len(vc)) - int(0.5 * len(vc))) * 0.2
        ax.plot(time_lags, vc, label=c)

    ax.set_xlabel("time lag (s)")
    ax.set_ylabel("AC(v)")
    ax.set_title(d)
    if i == 2:
        ax.legend()

# second row: firing rate distributions
for i, d in enumerate(drugs):

    ax = fig.add_subplot(grid[1, i*4:(i+1)*4])
    idx = multi_data["drug"] == d
    rates = multi_data

    for j, c in enumerate(["Control", "Amphetamine", "A + Drug"]):

        idx2 = multi_data["condition"] == c
        rates = multi_data["r"][idx & idx2]
        rates = np.mean(rates, axis=0)
        heights, bins, p = ax.hist(rates, bins=rate_bins, label=c, alpha=0.3, density=False)
        heights_f = gaussian_filter1d(heights, sigma=1)
        ax.plot(bins[1:], heights_f, color=p[0].get_facecolor(), linewidth=2.0, alpha=1.0)

    ax.set_xlabel("r (spikes/s)")
    ax.set_ylabel("count")
    ax.set_title(d)
    if i == 2:
        ax.legend()

# third row (single mouse: velocity dynamics and peak detection)
mouse_data = single_data["veh"]
mouse_behavior = mouse_data["behavior_data"]
ax = fig.add_subplot(grid[2, :])
start, stop = 2500, 3500
v = mouse_data["v_smooth"]
x = np.arange(len(v))*0.2
ax.plot(x[start:stop], v[start:stop])
pcols = {}
for i, b in enumerate(mouse_behavior["b"]):
    p = mouse_behavior["b_idx"][i]
    w = np.zeros_like(v)
    w[p] = 1.0
    area = ax.fill_between(x=x[start:stop], y1=0.0, y2=np.max(v[start:stop]), where=w[start:stop], label=b, alpha=0.6)
    pcols[b] = area.get_facecolor()
ax.legend()
ax.set_ylabel("v (cm/s)")
ax.set_xlabel("time (s)")
ax.set_title("mouse velocity")

# fourth row: velocity peak distributions
for i, drug in enumerate(drugs):

    # behavior distribution
    ax = fig.add_subplot(grid[3, i * 4:(i + 1) * 4])
    sb.barplot(df_sorted.loc[df_sorted.loc[:, "drug"] == drug, :],
                x="behavior", y="p(behavior)", hue="condition", ax=ax,
                errorbar="se", legend="auto" if i == 2 else False)
    ax.set_title(drug)

# fifth row: SPN dimensionality
for i, drug in enumerate(drugs):

    ax = fig.add_subplot(grid[4, i*4:(i+1)*4])
    sb.lineplot(df_sorted.loc[df_sorted.loc[:, "drug"] == drug, :],
                x="behavior", y="D(C)", hue="condition", style="SPNs", ax=ax,
                markers=True, dashes=True, err_style='bars', errorbar="se", legend="auto" if i == 2 else False)

# padding
fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, hspace=0.01, wspace=0.01)
plt.tight_layout()

# saving/plotting
fig.canvas.draw()
plt.savefig(f'{path}/spn_behavior_relationship2.svg', format="svg")
plt.show()
