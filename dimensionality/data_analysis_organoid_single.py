import numpy as np
from scipy.io import loadmat
import mat73
import matplotlib.pyplot as plt

path = "/home/richard-gast/Documents/data/organoid_dynamics"
file = "LFP_Sp_170316.mat"
try:
    data = loadmat(f"{path}/{file}", squeeze_me = False)
except NotImplementedError:
    data = mat73.loadmat(f"{path}/{file}")

lfp = data["LFP"]
spikes = data["spikes"]
time = np.squeeze(data["t_s"])
time_ds = np.squeeze(data["t_ds"])

wells = [4]

# loop over wells/organoids
for well in wells:

    lfp_well = lfp[well][0].T
    spikes_well = spikes[well]
    n_channels = lfp_well.shape[0]
    spikes_lfp = []

    # extract spikes and LFPs for particular well
    for i in range(n_channels):
        spike_times = np.zeros_like(time_ds)
        s_tmp = spikes_well[i]
        if s_tmp is not None and np.sum(s_tmp) > 0:
            s = np.asarray(s_tmp, dtype=np.int32) if s_tmp.shape else np.asarray([s_tmp], dtype=np.int32)
            for spike in s:
                idx = np.argmin(np.abs(time_ds -  time[spike]))
                spike_times[idx] = 1.0
        spikes_lfp.append(spike_times)
    spikes_lfp = np.asarray(spikes_lfp)

    # plot LFPs and derived spikes from single well
    fig, axes = plt.subplots(nrows=2, figsize=(12, 6))
    ax = axes[0]
    im = ax.imshow(lfp_well, interpolation="none", cmap="cividis", aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.7)
    ax.set_ylabel("channel")
    ax = axes[1]
    im = ax.imshow(spikes_lfp, interpolation="none", cmap="Greys", aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.7)
    ax.set_ylabel("channel")
    ax.set_xlabel("time steps")
    fig.suptitle(f"Organoid {well + 1}")
    plt.tight_layout()

plt.show()
