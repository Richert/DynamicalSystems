from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import pickle
import numpy as np


def get_hopf_area(hbs: np.ndarray, idx: int, cutoff: int):
    diff = np.diff(hbs)
    idx_l = hbs[idx]
    idx_r = idx_l
    while idx < len(diff):
        idx_r = hbs[idx]
        idx += 1
        if diff[idx-1] > cutoff:
            break
    return idx_l, idx_r, idx


# preparations
##############

# define condition
neuron_type = "rs"
path = "results"

# load data
data = pickle.load(open(f"{path}/spikes_{neuron_type}.pkl", "rb"))
I_ext = data["I_ext"]
delta = data["Delta"]
sd = data["SD"]

# analysis parameters
n_cycles = 5
hopf_start = 100
hopf_width = 30
hopf_height = 0.06
sigma_lowpass = 20
threshold = 0.12

# detect bifurcations in time series
####################################

for key in ["lorentz", "gauss"]:

    # filter data
    spikes = data[key]
    rates = np.mean(spikes, axis=1)
    filtered = gaussian_filter1d(rates, sigma=sigma_lowpass)

    # find fold bifurcation points
    indices = np.argwhere(filtered > threshold)
    if len(indices) > 1 and indices[-1] < len(filtered) - 1:
        idx_l = indices[0]
        idx_r = indices[-1]
        I_l = I_ext[idx_l]
        I_r = I_ext[idx_r]
        data[f"{key}_fold"] = {"idx": (idx_l, idx_r), "I_ext": (I_l, I_r), "delta": delta, "sd": sd}
    else:
        data[f"{key}_fold"] = {}

    # find hopf bifurcation points
    hbs, _ = find_peaks(filtered, width=hopf_width, prominence=hopf_height)
    hbs = hbs[hbs >= hopf_start]
    if len(hbs) > n_cycles:
        idx_l, idx_r = hbs[0], hbs[-1]
        I_r = I_ext[idx_r]
        I_l = I_ext[idx_l]
        data[f"{key}_hopf"] = {"idx": (idx_l, idx_r), "I_ext": (I_l, I_r), "delta": delta, "sd": sd}
    else:
        data[f"{key}_hopf"] = {}

# save results
##############

pickle.dump(data, open(f"{path}/spikes_{neuron_type}.pkl", "wb"))

# plotting
##########

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
markersize = 1
m = "x"
colors = ["black", "darkorange"]

# data parameters
neuron_ds = 10
bifurcation = "fold"
tau = 6.0

# create figure
fig = plt.figure(figsize=(12, 4))
grid = fig.add_gridspec(ncols=2, nrows=3, height_ratios=[2, 2, 1])

titles = [rf"Lorentzian heterogeneity with $\Delta_v = {np.round(delta, decimals=1)}$ mV",
          rf"Gaussian heterogeneity with $\sigma_v = {np.round(sd, decimals=1)}$ mV"]
time_windows = ([30000, 90000], [30000, 90000])
for idx, key in enumerate(["lorentz", "gauss"]):

    # plot spike raster
    ax1 = fig.add_subplot(grid[0, idx])
    spikes = data[key]
    x = np.linspace(0, spikes.shape[0]/10, spikes.shape[0])
    time = time_windows[idx]
    ax1.imshow(spikes[time[0]:time[1], ::neuron_ds].T, interpolation="antialiased", aspect="auto", cmap="Greys")
    ax1.set_xlabel("time (ms)")
    ax1.set_ylabel("neuron id")
    ax1.set_title(titles[idx])

    # plot average firing rate
    ax2 = fig.add_subplot(grid[1, idx])
    rates = np.mean(data[key], axis=1)*1e3/tau
    ax2.plot(x, rates, color="black")
    ax2.vlines(x=data[f"{key}_{bifurcation}"]["idx"][0]/10, ymin=0, ymax=np.max(rates), linestyle='dashed',
               color='slateblue')
    ax2.vlines(x=data[f"{key}_{bifurcation}"]["idx"][-1]/10, ymin=0, ymax=np.max(rates), linestyle='dashed',
               color='mediumorchid')
    ax2.vlines(x=time[0]/10, ymin=0, ymax=np.max(rates), linestyle='dashed',
               color='black')
    ax2.vlines(x=time[1]/10, ymin=0, ymax=np.max(rates), linestyle='dashed',
               color='black')
    ax2.set_xlabel("")
    ax2.set_ylabel(r"$r$ (Hz)")

    # plot input
    ax3 = fig.add_subplot(grid[2, idx])
    ax3.plot(x, I_ext, color="grey")
    ax3.vlines(x=data[f"{key}_{bifurcation}"]["idx"][0]/10, ymin=0, ymax=data[f"{key}_{bifurcation}"]["I_ext"][0],
               linestyle='dashed', color='slateblue')
    ax3.vlines(x=data[f"{key}_{bifurcation}"]["idx"][-1]/10, ymin=0, ymax=data[f"{key}_{bifurcation}"]["I_ext"][-1],
               linestyle='dashed', color='mediumorchid')
    ax3.hlines(y=data[f"{key}_{bifurcation}"]["I_ext"][0], xmin=0, xmax=data[f"{key}_{bifurcation}"]["idx"][0]/10,
               linestyle='dashed', color='slateblue')
    ax3.hlines(y=data[f"{key}_{bifurcation}"]["I_ext"][-1], xmin=0, xmax=data[f"{key}_{bifurcation}"]["idx"][-1]/10,
               linestyle='dashed', color='mediumorchid')
    ax3.set_xlabel("time (ms)")
    ax3.set_ylabel(r"$I_{ext}$ (pA)")

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/bifurcation_example_{neuron_type}.svg')
plt.show()
