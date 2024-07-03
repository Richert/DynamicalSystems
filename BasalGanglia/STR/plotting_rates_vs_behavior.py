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
sigma = 1
sr = 5
threshold = 0.1
threshold2 = 0.025

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
# plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams["pdf.use14corefonts"] = True

markersize = 6

# analysis
##########

results = {
    "condition": [], "moving": [], "turning": [], "time": [],
    "velocity": [], "acceleration": [], "angle": [], "angle_left": [], "angle_right": [],
    "rate_mean": [], "rate_change": [],
           }
for cond in ["amph", "veh"]:

    # extract data
    data_tmp = data.loc[data["condition"] == cond, :]
    rate_mean = data_tmp.loc[:, "rate_mean"]
    velocity = data_tmp.loc[:, "velocity"]
    angle = data_tmp.loc[:, "angle"]

    # calculate smooth variables
    rate_mean_smooth = gaussian_filter1d(rate_mean, sigma=sigma)
    velocity_smooth = gaussian_filter1d(velocity, sigma=sigma)
    angle_smooth = gaussian_filter1d(angle, sigma=sigma)
    rate_mean_smooth /= np.max(rate_mean_smooth)
    velocity_smooth /= np.max(velocity_smooth)

    # calculate changes in smoothed variables
    acceleration = np.diff(velocity_smooth, 1)
    acceleration /= np.max(acceleration)
    angle_change = np.diff(angle_smooth, 1)
    av_left = np.zeros_like(angle_change)
    av_left[angle_change > 0] = angle_change[angle_change > 0]
    av_left /= np.max(av_left)
    av_right = np.zeros_like(angle_change)
    av_right[angle_change < 0] = -angle_change[angle_change < 0]
    av_right /= np.max(av_right)
    rate_change = np.diff(rate_mean_smooth, 1)
    rate_change /= np.max(rate_change)
    n = len(acceleration)

    # find behavioral state
    moving = velocity_smooth[:n] > threshold
    left_turn = av_left > threshold2
    right_turn = av_right > threshold2
    turning = np.zeros_like(moving, dtype=np.int32)
    turning[left_turn] = 1
    turning[right_turn] = 2

    # save data
    results["condition"].extend([cond] * n)
    results["time"].extend(np.arange(0, n, dtype=np.float64) / sr)
    results["moving"].extend(moving[:n].tolist())
    results["turning"].extend(turning[:n].tolist())
    results["rate_mean"].extend(rate_mean_smooth[:n].tolist())
    results["rate_change"].extend(rate_change[:n].tolist())
    results["velocity"].extend(velocity_smooth[:n].tolist())
    results["acceleration"].extend(acceleration[:n].tolist())
    results["angle"].extend(angle_smooth[:n].tolist())
    results["angle_left"].extend(av_left[:n].tolist())
    results["angle_right"].extend(av_right[:n].tolist())

results = DataFrame.from_dict(results)

# plotting
##########

fig = plt.figure(figsize=(6, 5))
grid = fig.add_gridspec(nrows=4, ncols=2)
alpha = 0.5
titles = ["Vehicle condition", "Amphetamine condition"]
window = [200, 300]
state_colors = ["grey", "darkred", "royalblue"]
state_legend = [False, False, False]
state_labels = ["running", "left turn", "right turn"]
for i, cond in enumerate(["veh", "amph"]):

    # reduce data to condition
    results_tmp = results.loc[results["condition"] == cond, :]
    results_tmp = results_tmp.loc[(window[0] <= results_tmp.loc[:, "time"]) & (results_tmp.loc[:, "time"] < window[1])]
    turning = results_tmp.loc[:, "turning"].values
    moving = results_tmp.loc[:, "moving"].values
    x = results_tmp.loc[:, "time"].values

    # movement
    ax = fig.add_subplot(grid[i*2, :])
    ax.plot(x, results_tmp.loc[:, "velocity"], label="velocity", color=state_colors[0])
    ax.fill_between(x=x, y1=np.zeros_like(x), y2=results_tmp.loc[:, "velocity"], color=state_colors[0], alpha=alpha)
    ax.plot(x, results_tmp.loc[:, "angle_left"], label="angular vel. left", color=state_colors[1])
    ax.plot(x, results_tmp.loc[:, "angle_right"], label="angular vel. right", color=state_colors[2])
    ax.legend()
    ax.set_ylabel("velocity")
    ax.set_title(titles[i])

    # neural rates
    ax = fig.add_subplot(grid[i*2+1, :])
    mean_rate = results_tmp.loc[:, "rate_mean"]
    mean_rate /= np.max(mean_rate)
    ax.plot(x, mean_rate, color="black")
    ax.set_ylabel("event rate")
    ax.set_xlabel("time (s)")

    # add color-coding of turning states
    turn_borders = np.argwhere(np.abs(np.diff(turning, 1)) > 0.0).squeeze().tolist() + [len(turning) - 1]
    b0 = 0
    legend_objects = []
    legend_labels = []
    for b in turn_borders:
        idx = turning[b0 + 1]
        if idx != 0:
            obj = ax.fill_betweenx(y=[0.0, 1.0], x1=x[b0], x2=x[b], alpha=alpha, color=state_colors[idx])
            if state_legend[idx] is False:
                state_legend[idx] = True
                legend_objects.append(obj)
                legend_labels.append(state_labels[idx])
        b0 = b

    if i == 0:
        ax.legend(legend_objects, legend_labels)

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0.01, wspace=0.01)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'{path}/{drug}_{dose}_{mouse}_signal.pdf', format="pdf")
plt.show()
