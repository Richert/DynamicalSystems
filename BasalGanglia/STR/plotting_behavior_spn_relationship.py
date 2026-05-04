import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pandas import read_csv
import pickle
import sys

# preparations
##############

# load all-mice 1D data
condition = "p" #sys.argv[-1]
path = f"/home/rgast/data/parker_data"
df = read_csv(f"{path}/spn_behavior_{condition}.csv")
df.sort_values(["condition", "behavior", "SPNs"], inplace=True)

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

# post-analysis processing
idx = df.loc[:, "condition"] != "A + Low Dose"
df = df.loc[idx, :]
df.loc[df.loc[:, "condition"] == "A + High Dose", "condition"] = "A + Drug"

# data parameters
rate_bins = np.linspace(0.0, 0.5, num=20)

# plotting
##########

# plot settings
# import matplotlib.pyplot as plt
# import matplotlib
# import seaborn as sb
matplotlib.use("TkAgg")
plt.rcParams["font.size"] = 16.0
# plt.rcParams["axes.labelsize"] = 14.0
# plt.rcParams['xtick.labelsize'] = 14.0
# plt.rcParams['ytick.labelsize'] = 14.0
plt.rcParams["lines.markersize"] = 12.0
plt.rcParams["lines.linewidth"] = 2.0
import seaborn as sb
sb.set_palette("colorblind")

# figure layout
fig = plt.figure(figsize=(14, 12), layout="tight")
grid = fig.add_gridspec(nrows=4, ncols=12)

# first row (single mouse: velocity dynamics and peak detection)
mouse_data = single_data["veh"]
mouse_behavior = mouse_data["behavior_data"]
ax = fig.add_subplot(grid[0, :])
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

# second row: single mouse data
titles = ["Vehicle", "Amphetamine"]
b = "p2"
axes = [fig.add_subplot(grid[1, i*3:(i+1)*3]) for i in range(4)]
for i, c in enumerate(["veh", "amph"]):

    mouse_data = single_data[c]
    mouse_behavior = mouse_data["behavior_data"]
    idx = mouse_behavior["b"].index(b)

    # velocity autocorrelation for all behaviors
    n = len(mouse_data["v_c"])
    time_lags = (np.arange(n) - int(0.5*n))*0.2
    ax = axes[0]
    ax.plot(time_lags, mouse_data["v_c"], label=titles[i])
    # ax.legend()
    ax.set_ylabel("correlation")
    ax.set_xlabel("time lag (s)")
    ax.set_title("velocity AC (all behaviors)")

    # velocity autocorrelation for specified behavior
    ax = axes[1]
    ax.plot(time_lags, mouse_behavior["v_c"][idx], label=titles[i])
    # ax.legend()
    ax.set_ylabel("correlation")
    ax.set_xlabel("time lag")
    ax.set_title(f"velocity AC for {b}")

    # SPN firing rate distributions for all behaviors
    ax = axes[2]
    rates = np.mean(mouse_data["s_smooth"], axis=1)
    ax.hist(rates, bins=rate_bins, label=titles[i], alpha=0.5)
    # ax.legend()
    ax.set_ylabel("p(r)")
    ax.set_title(f"SPN rates (all behaviors)")
    ax.set_xlabel("r (spikes/s)")

    # SPN firing rate distirbutions for specified behavior
    ax = axes[3]
    r = mouse_behavior["r_dist"][idx]
    ax.hist(r, bins=rate_bins, label=titles[i], alpha=0.5)
    ax.legend()
    ax.set_ylabel("p(r)")
    ax.set_title(f"SPN rates for {b}")
    ax.set_xlabel("r (spikes/s)")

# all mice: line plots
for i, drug in enumerate(drugs):

    # behavior distribution
    ax = fig.add_subplot(grid[2, i * 4:(i + 1) * 4])
    sb.barplot(df.loc[df.loc[:, "drug"] == drug, :],
                x="behavior", y="p(behavior)", hue="condition", ax=ax,
                errorbar="se", legend="auto" if i == 2 else False)
    ax.set_title(drug)

    # firing rate heterogeneity
    ax = fig.add_subplot(grid[3, i*4:(i+1)*4])
    sb.lineplot(df.loc[df.loc[:, "drug"] == drug, :],
                x="behavior", y="D(r)", hue="condition", style="SPNs", ax=ax,
                markers=True, dashes=True, err_style='bars', errorbar="se", legend="auto" if i == 2 else False)

    # if i == 0 and j == 1:
    #     ax.set_title("statistics across all mice")

# padding
# fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, hspace=0.0, wspace=0.0)
plt.tight_layout()

# saving/plotting
fig.canvas.draw()
plt.savefig(f'{path}/spn_behavior_relationship.svg', format="svg")
plt.show()
