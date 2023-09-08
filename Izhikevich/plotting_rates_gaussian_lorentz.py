import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import pickle

#############
# load data #
#############

results = pickle.load(open("results/norm_lorentz_compare.pkl", "rb"))

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
fig = plt.figure(figsize=(12, 5))
grid = fig.add_gridspec(nrows=2, ncols=6)

# plot the firing rate distributions for the Lorentzian
for idx, Delta, rate_dist in zip([0, 3], results["Deltas"], results["lorentz"]["rate_dist"]):

    ax = fig.add_subplot(grid[0, idx])
    im = ax.imshow(np.log(np.asarray(rate_dist).T), cmap=plt.get_cmap('Reds'), aspect='auto', origin='lower')
    ax.set_xlabel(r"$I$ (pA)")
    ax.set_ylabel(r"$r_i$")
    ax.set_title(fr"$\Delta_v = {Delta}$")
    plt.colorbar(im, ax=ax, shrink=0.8)

# plot the firing rate distributions for the Gaussian
for idx, Delta, rate_dist in zip([0, 3], results["Deltas"], results["gauss"]["rate_dist"]):
    ax = fig.add_subplot(grid[1, idx])
    im = ax.imshow(np.log(np.asarray(rate_dist).T), cmap=plt.get_cmap('Reds'), aspect='auto', origin='lower')
    ax.set_xlabel(r"$I$ (pA)")
    ax.set_ylabel(r"$r_i$")
    ax.set_title(fr"$\sigma_v = {Delta}$")
    plt.colorbar(im, ax=ax, shrink=0.8)

# spiking raster plots for the Lorentzian
for idx, Delta, spikes in zip([(1, 3), (4, 6)], results["Deltas"], results["lorentz"]["spikes"]):
    ax = fig.add_subplot(grid[0, idx[0]:idx[1]])
    ax.imshow(spikes.T, interpolation="none", aspect="auto", cmap="Greys")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("neuron id")
    ax.set_title(fr"$\Delta_v = {Delta}$")

# spiking raster plots for the Gaussian
for idx, Delta, spikes in zip([(1, 3), (4, 6)], results["Deltas"], results["gauss"]["spikes"]):
    ax = fig.add_subplot(grid[1, idx[0]:idx[1]])
    ax.imshow(spikes.T, interpolation="none", aspect="auto", cmap="Greys")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("neuron id")
    ax.set_title(fr"$\sigma_v = {Delta}$")

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/lorentz_gauss_comparison.svg')
plt.show()
