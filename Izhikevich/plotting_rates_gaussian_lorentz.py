import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import seaborn as sb
import sys
import pickle
from scipy.signal import find_peaks
sys.path.append('../')

################
# preparations #
################

# choose neuron type
neuron_type = "rs"

# load data
results = pickle.load(open(f"results/norm_lorentz_{neuron_type}.pkl", "rb"))

# parameters
cutoff = 1000
n_bins = 20
x_ticks = 15
y_ticks = 40
fr_bins = np.linspace(0.0, 50.0, n_bins+1)
I_ext = np.round(np.linspace(50.0, 100.0, num=100), decimals=1)

############
# plotting #
############

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6

# create figure with grid
fig = plt.figure(figsize=(12, 6), layout="tight")
grid = GridSpec(nrows=4, ncols=4,
                width_ratios=[1.0, 0.7, 1.0, 0.7],
                height_ratios=[1.0, 0.7, 1.0, 0.7])

# plot the firing rate distributions for the uncoupled Lorentzian
for idx, Delta, rate_dist, rate_bins, cond in zip([0, 2], results["Deltas"], results["lorentz"]["rate_dist"],
                                                  results["lorentz"]["rate_bins"], ["A", "B"]):

    ax = plt.subplot(grid[0, idx])
    d = np.log(np.asarray(rate_dist))
    df = pd.DataFrame(d, index=I_ext, columns=np.round(rate_bins[1::], decimals=1))
    sb.heatmap(df, ax=ax, cmap=plt.get_cmap('Reds'), mask=d < -100.0, xticklabels=x_ticks, yticklabels=y_ticks)
    ax.set_ylabel(r"$I$ (pA)")
    ax.set_xlabel(r"$r_i$ (Hz)")
    ax.set_title(fr"({cond}) $\Delta_v = {Delta}$ (mV)")
    ax.invert_yaxis()
    # ax = plt.subplot(grid[0, idx+1])
    # Colorbar(mappable=im, ax=ax)

# plot the spike rate distribution for the coupled Lorentzian
for idx, Delta, spikes in zip([1, 3], results["Deltas"], results["lorentz"]["spikes"]):

    # identify spikes
    spike_rates = []
    for neuron in range(spikes.shape[1]):
        max_sig = np.max(spikes[:, neuron])
        if max_sig > 1e-4:
            signal = spikes[:, neuron] / max_sig
            peaks, _ = find_peaks(signal, height=0.5, width=5)
            spike_rates.append(len(peaks))
        else:
            spike_rates.append(0.0)

    # plot spike rate distribution
    ax = plt.subplot(grid[0, idx])
    ax.hist(spike_rates, bins=fr_bins, density=False, rwidth=0.75)
    ax.set_xlabel(r"$r_i$ (Hz)")
    ax.set_ylabel("neuron count")
    ax.set_title(fr"$\Delta_v = {Delta}$ (mV)")
    # ax.set_xlim([0, 40])

# plot the firing rate distributions for the Gaussian
for idx, SD, rate_dist, rate_bins, cond in zip([0, 2], results["SDs"], results["gauss"]["rate_dist"],
                                               results["gauss"]["rate_bins"], ["C", "D"]):
    ax = plt.subplot(grid[2, idx])
    d = np.log(np.asarray(rate_dist))
    df = pd.DataFrame(d, index=I_ext, columns=np.round(rate_bins[1::], decimals=1))
    sb.heatmap(df, ax=ax, cmap=plt.get_cmap('Reds'), mask=d < -100.0, xticklabels=x_ticks, yticklabels=y_ticks)
    ax.set_ylabel(r"$I$ (pA)")
    ax.set_xlabel(r"$r_i$ (Hz)")
    ax.set_title(fr"({cond}) $\sigma_v = {SD}$ (mV)")
    ax.invert_yaxis()

# plot the spike rate distribution for the coupled Gaussian
for idx, SD, spikes in zip([1, 3], results["SDs"], results["gauss"]["spikes"]):

    # identify spikes
    spike_rates = []
    for neuron in range(spikes.shape[1]):
        max_sig = np.max(spikes[:, neuron])
        if max_sig > 1e-4:
            signal = spikes[:, neuron] / max_sig
            peaks, _ = find_peaks(signal, height=0.5, width=5)
            spike_rates.append(len(peaks))
        else:
            spike_rates.append(0.0)

    # plot spike rate distribution
    ax = plt.subplot(grid[2, idx])
    ax.hist(spike_rates, bins=fr_bins, density=False, rwidth=0.75)
    ax.set_xlabel(r"$r_i$ (Hz)")
    ax.set_ylabel("neuron count")
    ax.set_title(fr"$\sigma_v = {SD}$ (mV)")
    # ax.set_xlim([0, 40])

# spiking raster plots for the Lorentzian
for idx, Delta, spikes in zip([(0, 2), (2, 4)], results["Deltas"], results["lorentz"]["spikes"]):
    ax = plt.subplot(grid[1, idx[0]:idx[1]])
    ax.imshow(spikes.T, interpolation="none", aspect="auto", cmap="Greys")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("neuron id")
    ax.set_title(fr"$\Delta_v = {Delta}$")

# spiking raster plots for the Gaussian
for idx, SD, spikes in zip([(0, 2), (2, 4)], results["SDs"], results["gauss"]["spikes"]):
    ax = plt.subplot(grid[3, idx[0]:idx[1]])
    ax.imshow(spikes.T, interpolation="none", aspect="auto", cmap="Greys")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("neuron id")
    ax.set_title(fr"$\sigma_v = {SD}$ (mV)")

# padding
fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/lorentz_gauss_{neuron_type}.svg', bbox_inches='tight')
plt.show()
