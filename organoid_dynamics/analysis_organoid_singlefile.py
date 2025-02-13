import multiprocessing as mp
import numpy as np
from scipy.io import loadmat
import os
from custom_functions import *
import matplotlib.pyplot as plt
import seaborn as sb

if __name__ == '__main__':

    mp.set_start_method("spawn")

    # data set specifics
    ####################

    # choose data set
    dataset_name = "trujilo_2019"
    path = f"/home/richard-gast/Documents/data/{dataset_name}"
    file = "170224"

    # set meta parameters
    wells = 8
    well_offset = 4
    sigma = 100
    bursting_width = 1000
    intraburst_width = 100
    bursting_height = 0.5
    intraburst_height = 0.3
    bursting_relheight = 0.9
    tau = 20e-3
    lfp_nperseg = 50000
    lfp_fmax = 59.8

    # data loading and processing
    #############################

    # prepare age calculation
    year_0 = 16
    month_0 = 7
    day_0 = 1

    # load data from file
    data = loadmat(f"{path}/LFP_Sp_{file}.mat", squeeze_me=False)

    # extract data
    lfp = data["LFP"]
    spike_times = data["spikes"]
    time = np.squeeze(data["t_s"])
    time_ds = np.round(np.squeeze(data["t_ds"]), decimals=4) # type: np.ndarray

    # calculate organoid age
    date = file.split(".")[0].split("_")[-1]
    year, month, day = int(date[:2]), int(date[2:4]), int(date[4:])
    month += int((year-year_0)*12)
    day += int((month-month_0)*30)
    age = day - day_0

    # calculate organoid statistics
    results_spikes = {
        "organoid": [], "dim": [], "spike_reg": [], "burst_freq": [], "burst_reg": [],
        "rate_avg": [], "rate_het": [], "intraburst_dim": [], "intraburst_spike_reg": [],
        "intraburst_rate_avg": [], "intraburst_rate_het": [], "intraburst_freq": []
    }
    results_lfps = {
        "organoid": [], "dim": [], "lfp_var": [], "freq": [], "psd": [], "max_freq": [], "max_pow": []
    }
    spikes = {}
    print(f"Starting to process organoid data for {date} (age = {age} days)")
    with mp.Pool(processes=wells) as pool:

        # loop over wells/organoids
        res_lfp = [pool.apply_async(organoid_lfp_analysis,
                                    (well, age, lfp[well + well_offset], time_ds, lfp_nperseg, lfp_fmax))
                   for well in range(wells)]
        res_spikes = [pool.apply_async(organoid_spike_analysis,
                                       (well, age, spike_times[well + well_offset], time, time_ds,
                                        sigma, bursting_width, intraburst_width, bursting_height, intraburst_height,
                                        bursting_relheight, tau, True))
                      for well in range(wells)]

        # save spiking results
        for r in res_spikes:
            res_tmp = r.get()
            for key in results_spikes.keys():
                results_spikes[key].append(res_tmp[key])
            spikes[res_tmp["organoid"]] = res_tmp["spikes"]

        # save LFP results
        for r in res_lfp:
            res_tmp = r.get()
            for key in results_lfps.keys():
                results_lfps[key].append(res_tmp[key])

    print(f"Finished processing all organoids from {date}.")

    # plot results
    ##############

    # bar graph of spiking summary statistics
    keys = ["dim", "rate_avg", "rate_het", "burst_freq", "burst_reg", "intraburst_dim", "intraburst_freq"]
    _, ax = plt.subplots(figsize=(12, 5))
    ax.violinplot([results_spikes[key] for key in keys], showmedians=True)
    ax.set_xticks(np.arange(len(keys)) + 1, labels=keys)
    ax.set_title(f"Spiking statistics for organoids at {date} (age = {age} days)")
    plt.tight_layout()

    # bar graphs for lfp summary statistics
    keys = ["dim", "lfp_var", "max_freq", "max_pow"]
    _, ax = plt.subplots(figsize=(12, 5))
    ax.violinplot([results_lfps[key] for key in keys], showmedians=True)
    ax.set_xticks(np.arange(len(keys)) + 1, labels=keys)
    ax.set_title(f"LFP statistics for organoids at {date} (age = {age} days)")
    plt.tight_layout()

    # PSD plots
    f = results_lfps["freq"][0]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.asarray(results_lfps["freq"]).T, np.asarray(results_lfps["psd"]).T)
    ax.legend([f"well {i+1}" for i in range(wells)])
    ax.set_ylabel("log(psd)")
    ax.set_xlabel("frequency (Hz)")
    ax.set_title("Detrended PSD")
    plt.tight_layout()

    # example spike patterns
    organoids = np.arange(wells)
    for well in organoids:
        _, axes = plt.subplots(nrows=2, figsize=(12, 6))
        ax = axes[0]
        ax.imshow(spikes[well], aspect="auto", interpolation="none", cmap="Greys")
        ax.set_xlabel("time")
        ax.set_ylabel("channels")
        ax.set_title(f"Spiking dynamics for organoid {well+1}")
        ax = axes[1]
        ax.plot(time_ds, 1e3*np.mean(spikes[well], axis=0))
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("firing rate (Hz)")
    plt.tight_layout()

    plt.show()
