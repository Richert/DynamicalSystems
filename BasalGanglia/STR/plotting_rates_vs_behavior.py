import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from pandas import read_csv, DataFrame

# preparations
##############

# choose condition
drug = "SKF38393"
dose = "Vehicle"
mouse = "m120"

# meta parameters
sigma_movement = 20
sigma_rates = 2
sr = 5

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
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6

# analysis
##########

results = {"condition": [], "velocity": [], "acceleration": [], "angle": [], "angle_change": [],
           "rate_mean": [], "rate_std": [], "time": []}
for cond in ["amph", "veh"]:

    # extract data
    data_tmp = data.loc[data["condition"] == cond, :]
    rate_mean = data_tmp.loc[:, "rate_mean"]
    rate_std = data_tmp.loc[:, "rate_std"]
    velocity = data_tmp.loc[:, "velocity"]
    angle = data_tmp.loc[:, "angle"]

    # calculate smooth variables
    rate_mean_smooth = gaussian_filter1d(rate_mean, sigma=sigma_rates)
    rate_std_smooth = gaussian_filter1d(rate_std, sigma=sigma_rates)
    velocity_smooth = gaussian_filter1d(velocity, sigma=sigma_movement)
    angle_smooth = gaussian_filter1d(angle, sigma=sigma_movement)

    # calculate change in velocity and angle
    acceleration = np.diff(velocity_smooth, 1)
    acceleration /= np.max(acceleration)
    angle_change = np.diff(angle_smooth, 1)
    angle_change /= np.max(angle_change)

    # save data
    n = len(acceleration)
    results["condition"].extend([cond] * n)
    results["time"].extend(np.arange(0, n, dtype=np.float64) / sr)
    results["rate_mean"].extend(rate_mean_smooth[:n].tolist())
    results["rate_std"].extend(rate_std_smooth[:n].tolist())
    results["velocity"].extend(velocity_smooth[:n].tolist())
    results["acceleration"].extend(acceleration[:n].tolist())
    results["angle"].extend(angle_smooth[:n].tolist())
    results["angle_change"].extend(angle_change[:n].tolist())

results = DataFrame.from_dict(results)

# plotting
##########

fig = plt.figure(figsize=(12, 6))
grid = fig.add_gridspec(nrows=3, ncols=2)
titles = ["Control condition", "Dopamine condition"]
window = [200, 600]
c1 = "black"
c2 = "darkred"
for i, cond in enumerate(["veh", "amph"]):

    # reduce data to condition
    results_tmp = results.loc[results["condition"] == cond, :]
    results_tmp = results_tmp.loc[(window[0] <= results_tmp.loc[:, "time"]) & (results_tmp.loc[:, "time"] < window[1])]
    x = results_tmp.loc[:, "time"]

    # movement
    ax = fig.add_subplot(grid[0, i])
    ax.plot(x, results_tmp.loc[:, "velocity"], color=c1)
    ax2 = ax.twinx()
    ax2.plot(x, results_tmp.loc[:, "acceleration"], color=c2)
    if i == 0:
        ax.set_ylabel("velocity", color=c1)
    else:
        ax2.set_ylabel("acceleration", color=c2)
    ax.set_title(titles[i])

    # angles
    ax = fig.add_subplot(grid[1, i])
    ax.plot(x, results_tmp.loc[:, "angle"], color=c1)
    ax2 = ax.twinx()
    ax2.plot(x, results_tmp.loc[:, "angle_change"], color=c2)
    if i == 0:
        ax.set_ylabel("angle", color=c1)
    else:
        ax2.set_ylabel("angle change", color=c2)

    # neural rates
    ax = fig.add_subplot(grid[2, i])
    mean_rate = results_tmp.loc[:, "rate_mean"]
    v = results_tmp.loc[:, "rate_std"]
    lower_bound = mean_rate - 0.25*v
    lower_bound[lower_bound < 0.0] = 0.0
    upper_bound = mean_rate + 0.25*v
    ax.plot(x, mean_rate, color=c1, label="mean rate")
    # ax.plot(avg_sr[:-1] * (smoothed_speed[:-1] >= speed_threshold) * (speed_diff > acceleration_threshold), label="accelerating")
    # ax.plot(avg_sr[:-1] * (smoothed_speed[:-1] >= speed_threshold) * (speed_diff < -acceleration_threshold), label="decelerating")
    ax.fill_between(x=x, y1=lower_bound, y2=upper_bound, alpha=0.3, color=c1, linewidth=0.0)
    # ax.legend()
    if i == 0:
        ax.set_ylabel("MSN spike rate")
    ax.set_xlabel("time (s)")

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0.01, wspace=0.01)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'{path}/{drug}_{dose}_{mouse}_signal.pdf')
plt.show()
