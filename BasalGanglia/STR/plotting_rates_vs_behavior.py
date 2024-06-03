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
sigma1 = 2
sigma2 = 2
sr = 5
threshold = 0.1
threshold2 = 0.05

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
plt.rcParams['lines.linewidth'] = 1.2
markersize = 6

# analysis
##########

results = {
    "condition": [], "state": [], "time": [],
    "velocity": [], "acceleration": [], "angle": [], "angle_change": [], "rate_mean": [], "rate_change": [],
    "velocity2": [], "acceleration2": [], "angle2": [], "angle_change2": [], "rate_mean2": [], "rate_change2": [],
           }
for cond in ["amph", "veh"]:

    # extract data
    data_tmp = data.loc[data["condition"] == cond, :]
    rate_mean = data_tmp.loc[:, "rate_mean"]
    velocity = data_tmp.loc[:, "velocity"]
    angle = data_tmp.loc[:, "angle"]

    # calculate smooth variables
    rate_mean_smooth = gaussian_filter1d(rate_mean, sigma=sigma1)
    velocity_smooth = gaussian_filter1d(velocity, sigma=sigma1)
    angle_smooth = gaussian_filter1d(angle, sigma=sigma1)
    rate_mean_smooth /= np.max(rate_mean_smooth)
    velocity_smooth /= np.max(velocity_smooth)

    # calculate changes in smoothed variables
    acceleration = np.diff(velocity_smooth, 1)
    acceleration /= np.max(acceleration)
    angle_change = np.abs(np.diff(angle_smooth, 1))
    angle_change /= np.max(angle_change)
    rate_change = np.diff(rate_mean_smooth, 1)
    rate_change /= np.max(rate_change)
    n = len(acceleration)

    # find behavioral state
    moving = velocity_smooth[:n] > threshold
    turning = (angle_change > threshold2) & ~moving
    state = np.zeros_like(moving, dtype=np.int32)
    state[moving] = 1
    state[turning] = 2

    # get less smooth version of signals for plotting
    rate_mean_smooth2 = gaussian_filter1d(rate_mean, sigma=sigma2)
    velocity_smooth2 = gaussian_filter1d(velocity, sigma=sigma2)
    angle_smooth2 = gaussian_filter1d(angle, sigma=sigma2)
    rate_mean_smooth2 /= np.max(rate_mean_smooth2)
    velocity_smooth2 /= np.max(velocity_smooth2)
    acceleration2 = np.diff(velocity_smooth2, 1)
    acceleration2 /= np.max(acceleration2)
    angle_change2 = np.abs(np.diff(angle_smooth2, 1))
    angle_change2 /= np.max(angle_change2)
    rate_change2 = np.diff(rate_mean_smooth2, 1)
    rate_change2 /= np.max(rate_change2)

    # save data
    results["condition"].extend([cond] * n)
    results["time"].extend(np.arange(0, n, dtype=np.float64) / sr)
    results["state"].extend(state[:n].tolist())
    results["rate_mean"].extend(rate_mean_smooth[:n].tolist())
    results["rate_change"].extend(rate_change[:n].tolist())
    results["velocity"].extend(velocity_smooth[:n].tolist())
    results["acceleration"].extend(acceleration[:n].tolist())
    results["angle"].extend(angle_smooth[:n].tolist())
    results["angle_change"].extend(angle_change[:n].tolist())
    results["rate_mean2"].extend(rate_mean_smooth2[:n].tolist())
    results["rate_change2"].extend(rate_change2[:n].tolist())
    results["velocity2"].extend(velocity_smooth2[:n].tolist())
    results["acceleration2"].extend(acceleration2[:n].tolist())
    results["angle2"].extend(angle_smooth2[:n].tolist())
    results["angle_change2"].extend(angle_change2[:n].tolist())

results = DataFrame.from_dict(results)

# plotting
##########

fig = plt.figure(figsize=(6, 7))
grid = fig.add_gridspec(nrows=5, ncols=2)
alpha = 0.5
titles = ["Control condition", "Dopamine condition"]
window = [200, 300]
state_colors = ["grey", "darkred"]
state_legend = [False, False, False]
state_labels = ["move", "turn"]
for i, cond in enumerate(["veh", "amph"]):

    # reduce data to condition
    results_tmp = results.loc[results["condition"] == cond, :]
    results_tmp = results_tmp.loc[(window[0] <= results_tmp.loc[:, "time"]) & (results_tmp.loc[:, "time"] < window[1])]
    state = results_tmp.loc[:, "state"].values
    x = results_tmp.loc[:, "time"].values

    # movement
    ax = fig.add_subplot(grid[i*2, :])
    ax.plot(x, results_tmp.loc[:, "velocity2"], label="velocity", color=state_colors[0])
    ax.fill_between(x=x, y1=np.zeros_like(x), y2=results_tmp.loc[:, "velocity2"], color=state_colors[0], alpha=alpha)
    ax.plot(x, results_tmp.loc[:, "angle_change2"], label="angular velocity", color=state_colors[1])
    ax.legend()
    ax.set_title(titles[i])

    # neural rates
    ax = fig.add_subplot(grid[i*2+1, :])
    mean_rate = results_tmp.loc[:, "rate_mean2"]
    mean_rate /= np.max(mean_rate)
    ax.plot(x, mean_rate, color="black")
    borders = np.argwhere(np.abs(np.diff(state, 1)) > 0.0).squeeze().tolist() + [len(state)-1]
    b0 = 0
    legend_objects = []
    legend_labels = []
    for b in borders:
        idx = state[b0+1]
        if idx != 0:
            obj = ax.fill_betweenx(y=[0.0, 1.0], x1=x[b0], x2=x[b], alpha=alpha, color=state_colors[idx-1])
            if state_legend[idx-1] is False:
                state_legend[idx-1] = True
                legend_objects.append(obj)
                legend_labels.append(state_labels[idx-1])
        b0 = b
    if i == 0:
        ax.legend(legend_objects, legend_labels)
    ax.set_ylabel("FR")
    ax.set_xlabel("time (s)")

# bar graph: spike rates
ax = fig.add_subplot(grid[4, 0])
sns.barplot(results, x="state", y="rate_mean", hue="condition", ax=ax, palette="dark")
ax.set_xlabel("behavioral state")
ax.set_ylabel("FR")
ax.set_xticks([0, 1, 2], labels=["rest", "move", "turn"])

# bar graph: rate change
ax = fig.add_subplot(grid[4, 1])
sns.barplot(results, x="state", y="rate_change", hue="condition", ax=ax, palette="dark")
ax.set_xlabel("behavioral state")
ax.set_ylabel("FR change")
ax.set_xticks([0, 1, 2], labels=["rest", "move", "turn"])

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0.01, wspace=0.01)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'{path}/{drug}_{dose}_{mouse}_signal.svg')
plt.show()
