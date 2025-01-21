import numpy as np
from scipy.io import loadmat
import mat73
import os
from numba import prange
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from custom_functions import *
import matplotlib.pyplot as plt

# data set specifics
####################

path = "/media/richard/data/trujilo_2019"
wells = 8
well_offset = 4
bursting_sigma = 100
bursting_dist = 100

# data loading and processing
#############################

# prepare results data structure
results = {
    "organoid": [], "age": [], "dim": [], "isi_mean": [], "isi_std": [], "ibi_mean": [], "ibi_std": [],
    "fr_mean": [], "fr_std": [], "intraburst_dim": [], "intraburst_isi_mean": [], "intraburst_isi_std": [],
    "intraburst_fr_mean": [], "intraburst_fr_std": []
}

# prepare age calculation
year_old = 0
month_old = 0
day_old = 0

for file in os.listdir(path):

    # load data from file
    try:
        data = loadmat(f"{path}/{file}", squeeze_me = False)
    except NotImplementedError:
        data = mat73.loadmat(f"{path}/{file}")

    # extract data
    lfp = data["LFP"]
    spike_times = data["spikes"]
    time = np.squeeze(data["t_s"])
    time_ds = np.round(np.squeeze(data["t_ds"]), decimals=3) # type: np.ndarray
    xticks = np.arange(0, stop=time_ds.shape[0], step=50000, dtype=np.int32)

    # calculate organoid age
    date = file.split(".")[0].split("_")[-1]
    year, month, day = int(date[:2]), int(date[2:4]), int(date[4:])
    if len(results["age"]) < 1:
        age = 0
        year_old, month_old, day_old = year, month, day
    else:
        if year > year_old:
            month += 12
        if month > month_old:
            day += 30
        age = day - day_old
        year_old, month_old, day_old = year, month, day

    # loop over wells/organoids
    for well in prange(wells):

        # save condition data
        results["organoid"].append(well)
        results["age"].append(age)

        # get data from well
        lfp_well = lfp[well + well_offset][0].T
        spikes_well = spike_times[well + well_offset]
        n_channels = lfp_well.shape[0]

        # extract spikes and calculate firing rate statistics
        spikes = extract_spikes(n_channels, time_ds, spikes_well)
        fr = np.mean(spikes, axis=0)
        results["fr_mean"].append(np.mean(fr))
        results["fr_std"].append(np.mean(np.std(spikes, axis=0)))

        # calculate global dimensionality
        results["dim"].append(get_dim(spikes, center=True))

        # calculate global inter-spike interval statistics
        isi = inter_spike_intervals(spikes)
        results["isi_mean"].append(np.mean(isi))
        results["isi_std"].append(np.std(isi)/np.mean(isi))

        # calculate bursting frequency
        fr_smoothed = gaussian_filter1d(fr, sigma=bursting_sigma)
        fr_peaks, peak_props = find_peaks(fr_smoothed, distance=bursting_dist)
        ibi = np.diff(fr_peaks)
        results["ibi_mean"].append(np.mean(ibi))
        results["ibi_std"].append(np.std(ibi)/np.mean(ibi))

        # plotting
        fig, axes = plt.subplots(nrows=3, figsize=(12, 9))
        ax = axes[0]
        ax.plot(time_ds, fr, label="raw")
        ax.plot(time_ds, fr_smoothed, label="smoothed")
        for peak in fr_peaks:
            ax.axvline(x=time_ds[peak], color="black", linestyle="dashed")
        ax.set_ylabel("rate")
        ax.set_xlabel("time")
        ax.legend()
        ax = axes[1]
        ax.imshow(spikes, aspect="auto", interpolation="none", cmap="Greys")
        ax.set_ylabel("neurons")
        ax.set_xlabel("steps")
        ax = axes[2]
        ax.histogram(isi)
        ax.set_xlabel("ISI")
        ax.set_ylabel("count")
        plt.tight_layout()
        plt.show()

    # plot LFPs and derived spikes from single well
    fig, axes = plt.subplots(nrows=2, figsize=(12, 6))
    ax = axes[0]
    im = ax.imshow(lfp_well, interpolation="none", cmap="cividis", aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.7)
    ax.set_ylabel("channel")
    ax.set_xticks(xticks, labels=time_ds[xticks])
    ax = axes[1]
    im = ax.imshow(spikes_lfp, interpolation="none", cmap="Greys", aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.7)
    ax.set_xticks(xticks, labels=time_ds[xticks])
    ax.set_ylabel("channel")
    ax.set_xlabel("time")
    fig.suptitle(f"Organoid {well + 1}")
    plt.tight_layout()

plt.show()
