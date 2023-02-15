import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import welch, coherence
import pickle
import numpy as np
import sys


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


def minmax(x: np.ndarray) -> np.ndarray:
    x -= np.min(x)
    x /= np.max(x)
    return x


# conditions to loop over
#########################

# define conditions
fn = sys.argv[-1]  #"rs_arnold_tongue"
conditions = ["het", "hom"]
titles = [r"$\Delta = 1.0$", r"$\Delta = 0.1$"]
base_len = 6
fig = plt.figure(1, figsize=(int(len(conditions)*base_len), base_len))
grid = GridSpec(ncols=len(conditions), nrows=1, figure=fig)

# analysis meta parameters
nps = 16000
window = 'hamming'
epsilon = 1e-1

for idx, (cond, title) in enumerate(zip(conditions, titles)):

    # load data
    ###########

    data = pickle.load(open(f"{fn}_{cond}.pkl", "rb"))
    alphas = data["alphas"]
    omegas = data["omegas"]
    res = data["res"]
    res_map = data["map"]
    dts = res.index[1] - res.index[0]

    # calculate entrainment
    #######################

    # calculate and store entrainment
    entrainment = np.zeros((len(alphas), len(omegas)))
    for key in res_map.index:

        # extract parameter set
        omega = res_map.at[key, 'omega']
        alpha = res_map.at[key, 'alpha']

        # calculate psd of firing rate dynamics
        freqs, pows = welch(data["res"]["ik"][key].values.squeeze(), fs=1/dts, window=window, nperseg=nps)
        # sig = minmax(data["res"]["ik"][key].values.squeeze())
        # driver = minmax(data["res"]["ko"][key].values.squeeze())
        # freqs, coh = coherence(sig, driver, fs=1/dts, window=window, nperseg=nps, axis=0)

        # fig, axes = plt.subplots(nrows=3, figsize=(12, 6))
        # ax = axes[0]
        # ax.plot(data["res"]["ik"][key]*1e3)
        # ax.set_xlabel("time (ms)")
        # ax.set_ylabel("r (Hz)")
        # ax = axes[1]
        # ax.plot(np.sin(2*np.pi*data["res"]["ko"][key]))
        # ax.set_xlabel("time (ms)")
        # ax.set_ylabel("inp")
        # ax = axes[2]
        # ax.plot(freqs[freqs < 0.1]*1e3, pows[freqs < 0.1])
        # ax.set_xlabel("f (Hz)")
        # ax.set_ylabel("PSD")
        # plt.show()

        # find coherence matrix position that corresponds to these parameters
        idx_r = np.argmin(np.abs(alphas - alpha))
        idx_c = np.argmin(np.abs(omegas - omega))

        # store coherence value at driving frequency
        entrainment[idx_r, idx_c] = pows[np.argmin(np.abs(freqs-omega))]/np.max(pows)  # coh[np.argmin(np.abs(freqs-omega))]

    # plot entrainment
    ##################

    ax = fig.add_subplot(grid[0, idx])
    cax = ax.imshow(entrainment[::-1, :], aspect='equal', interpolation="none")
    ax.set_xlabel(r'$\omega$ (Hz)')
    ax.set_ylabel(r'$\alpha$ (Hz)')
    ax.set_xticks(np.arange(0, len(omegas), 3))
    ax.set_yticks(np.arange(0, len(alphas), 3))
    ax.set_xticklabels(np.round(omegas[::3]*1e3, decimals=1))
    ax.set_yticklabels(np.round(alphas[::-3]*1e3, decimals=1))
    plt.title(f"Entrainment for {title}")

plt.colorbar(cax, ax=ax, shrink=0.5)

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/rs_arnold_tongue.pdf')
plt.show()
