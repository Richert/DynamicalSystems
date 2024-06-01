import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from pandas import read_csv, DataFrame

# preparations
##############

# choose condition
drug = "SKF38393"
dose = "Vehicle"
mouse = "m106"

# meta parameters
sigma_movement = 20
sigma_rates = 20
sr = 5
threshold = 0.02
threshold2 = 0.08
threshold3 = 0.08

# load data
path = "/run/user/1000/gvfs/smb-share:server=fsmresfiles.fsm.northwestern.edu,share=fsmresfiles/Basic_Sciences/Phys/Kennedylab/Parkerlab/Richard"
data = read_csv(f"{path}/data/{drug}_{dose}_combined.csv")
data = data.loc[data["mouse"] == mouse, :]

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 12.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.5
markersize = 6

# analysis
##########

results = {"condition": [], "velocity": [], "acceleration": [], "angle": [], "angle_change": [],
           "rate_mean": [], "rate_change": [], "state": [], "time": []}
for cond in ["amph", "veh"]:

    # extract data
    data_tmp = data.loc[data["condition"] == cond, :]
    rate_mean = data_tmp.loc[:, "rate_mean"]
    velocity = data_tmp.loc[:, "velocity"]
    angle = data_tmp.loc[:, "angle"]

    # calculate smooth variables
    rate_mean_smooth = gaussian_filter1d(rate_mean, sigma=sigma_rates)
    velocity_smooth = gaussian_filter1d(velocity, sigma=sigma_movement)
    angle_smooth = gaussian_filter1d(angle, sigma=sigma_movement)
    rate_mean_smooth /= np.max(rate_mean_smooth)
    velocity_smooth /= np.max(velocity_smooth)

    # calculate changes in smoothed variables
    acceleration = np.diff(velocity_smooth, 1)
    acceleration /= np.max(acceleration)
    angle_change = np.diff(angle_smooth, 1)
    angle_change /= np.max(angle_change)
    rate_change = np.diff(rate_mean_smooth, 1)
    rate_change /= np.max(rate_change)
    acceleration[acceleration > 0] = np.sqrt(acceleration[acceleration > 0])
    acceleration[acceleration < 0] = -np.sqrt(-acceleration[acceleration < 0])
    angle_change[angle_change > 0] = np.sqrt(angle_change[angle_change > 0])
    angle_change[angle_change < 0] = -np.sqrt(-angle_change[angle_change < 0])
    n = len(acceleration)

    # find behavioral state
    rest = velocity_smooth[:n] < threshold
    decelerating = acceleration < -threshold2
    accelerating = acceleration > threshold3
    state = np.zeros_like(rest, dtype=np.int32)
    state[rest] = 1
    state[decelerating] = 2
    state[accelerating] = 3

    # save data
    results["condition"].extend([cond] * n)
    results["time"].extend(np.arange(0, n, dtype=np.float64) / sr)
    results["rate_mean"].extend(rate_mean_smooth[:n].tolist())
    results["rate_change"].extend(rate_change[:n].tolist())
    results["velocity"].extend(velocity_smooth[:n].tolist())
    results["acceleration"].extend(acceleration[:n].tolist())
    results["angle"].extend(angle_smooth[:n].tolist())
    results["angle_change"].extend(angle_change[:n].tolist())
    results["state"].extend(state[:n].tolist())

results = DataFrame.from_dict(results)

# plotting
##########

fig = plt.figure(figsize=(6, 7))
grid = fig.add_gridspec(nrows=5, ncols=2)
titles = ["Control condition", "Dopamine condition"]
window = [200, 500]
state_colors = ["black", "darkred", "darkgreen"]
c1 = "black"
c2 = "darkred"
for i, cond in enumerate(["veh", "amph"]):

    # reduce data to condition
    results_tmp = results.loc[results["condition"] == cond, :]
    results_tmp = results_tmp.loc[(window[0] <= results_tmp.loc[:, "time"]) & (results_tmp.loc[:, "time"] < window[1])]
    state = results_tmp.loc[:, "state"].values
    x = results_tmp.loc[:, "time"].values

    # movement
    ax = fig.add_subplot(grid[i*2, :])
    ax.plot(x, results_tmp.loc[:, "acceleration"], color=c1)
    ax2 = ax.twinx()
    ax2.plot(x, results_tmp.loc[:, "angle_change"], color=c2)
    if i == 0:
        ax.set_ylabel("acceleration", color=c1)
    else:
        ax2.set_ylabel("HNT change", color=c2)
    ax.set_title(titles[i])

    # neural rates
    ax = fig.add_subplot(grid[i*2+1, :])
    mean_rate = results_tmp.loc[:, "rate_mean"]
    mean_rate /= np.max(mean_rate)
    ax.plot(x, mean_rate, color=c1)
    ax2 = ax.twinx()
    ax2.plot(x, results_tmp.loc[:, "rate_change"], color=c2)
    borders = np.argwhere(np.abs(np.diff(state, 1)) > 0.0).squeeze().tolist() + [len(state)-1]
    b0 = 0
    for b in borders:
        idx = state[b0+1]
        if idx != 0:
            ax.fill_betweenx(y=[0.0, 1.0], x1=x[b0], x2=x[b], alpha=0.3, color=state_colors[idx-1])
        b0 = b
    # ax.legend()
    if i == 0:
        ax.set_ylabel("spike rate (SR)")
    else:
        ax2.set_ylabel("SR change", color=c2)
    ax.set_xlabel("time (s)")

# bar graph: spike rates
ax = fig.add_subplot(grid[4, 0])
sns.barplot(results, x="state", y="rate_mean", hue="condition", ax=ax, palette="dark")
ax.set_xlabel("behavioral state")
ax.set_ylabel("SR")
ax.set_xlim([0.5, 3.5])
ax.set_xticks([1, 2, 3], labels=["rest", "dec.", "acc."])

# bar graph: rate change
ax = fig.add_subplot(grid[4, 1])
sns.barplot(results, x="state", y="rate_change", hue="condition", ax=ax, palette="dark")
ax.set_xlabel("behavioral state")
ax.set_ylabel("SR change")
ax.set_xlim([0.5, 3.5])
ax.set_xticks([1, 2, 3], labels=["rest", "dec.", "acc."])

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0.01, wspace=0.01)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'{path}/{drug}_{dose}_{mouse}_signal.svg')
plt.show()
