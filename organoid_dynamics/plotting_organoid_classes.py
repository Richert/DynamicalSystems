import multiprocessing as mp
import numpy as np
from scipy.io import loadmat
import os
from custom_functions import *
import matplotlib.pyplot as plt
import seaborn as sb


def organoid_analysis(path: str, file: str, well: int, tau: float, sigma: int, burst_width: int, burst_height: float,
                      burst_sep: int, width_at_height: float, waveform_length: int) -> dict:

    # prepare age calculation
    year_0 = 16
    month_0 = 7
    day_0 = 1

    # load data from file
    data = loadmat(f"{path}/LFP_Sp_{file}.mat", squeeze_me=False)

    # extract data
    spike_times = data["spikes"]
    time = np.squeeze(data["t_s"])
    time_ds = np.round(np.squeeze(data["t_ds"]), decimals=4)  # type: np.ndarray

    # calculate organoid age
    date = file.split(".")[0].split("_")[-1]
    year, month, day = int(date[:2]), int(date[2:4]), int(date[4:])
    month += int((year - year_0) * 12)
    day += int((month - month_0) * 30)
    age = day - day_0

    # calculate firing rate
    dts = float(time_ds[1] - time_ds[0]) * 1e3
    spikes = extract_spikes(time, time_ds, spike_times[well - 1])
    spikes_smoothed = convolve_exp(spikes, tau=tau, dt=dts, normalize=False)
    fr = np.mean(spikes_smoothed, axis=0) * 1e3 / tau

    # get bursting stats
    res = get_bursting_stats(fr, sigma=sigma, burst_width=burst_width, rel_burst_height=burst_height,
                             burst_sep=burst_sep, width_at_height=width_at_height, waveform_length=waveform_length)
    res["age"] = age
    res["file"] = file
    res["organoid"] = well
    return res


if __name__ == '__main__':

    mp.set_start_method("spawn")

    # data set specifics
    ####################

    # choose data to plot
    dataset_name = "trujilo_2019"
    path = f"/home/richard-gast/Documents/data/{dataset_name}"
    files = ["160803", "161001", "161110", "170203"]
    wells = [6, 6, 6, 6]

    # set meta parameters
    well_offset = 4
    tau = 10.0
    sigma = 20
    burst_width = 100
    burst_height = 0.6
    burst_sep = 1000
    burst_relheight = 0.9
    n_bins = 10
    waveform_length = 3000

    # calculate organoid statistics
    results = {"organoid": [], "age": [], "file": [], "ibi": [], "waveform_mean": [], "waveform_std": [], "fr": []}

    print(f"Starting to process organoid data")
    with mp.Pool(processes=len(files)) as pool:

        # loop over wells/organoids
        res = [pool.apply_async(organoid_analysis,
                                (path, file, well + well_offset, tau, sigma, burst_width, burst_height, burst_sep,
                                 burst_relheight, waveform_length))
                      for file, well in zip(files, wells)]
        # res = [organoid_analysis(path, file, well + well_offset, tau, sigma, burst_width, burst_height, burst_relheight, waveform_length)
        #        for file, well in zip(files, wells)]

        # save results
        for r in res:
            res_tmp = r.get()
            for key in results.keys():
                results[key].append(res_tmp[key])

    print(f"Finished processing all organoids.")

    # plot results
    ##############

    for i in range(len(files)):

        fig = plt.figure(figsize=(12, 9))
        grid = fig.add_gridspec(nrows=2, ncols=2)
        fig.suptitle(f"Dynamics of organoid {results['organoid'][i]} at date {results['file'][i]} (age = {results['age'][i]})")

        # firing rate dynamics
        ax = fig.add_subplot(grid[0, :])
        ax.plot(results["fr"][i])
        ax.set_xlabel("time")
        ax.set_ylabel("firing rate")
        ax.set_title("average firing rate dynamics")

        # ibi distribution
        ax = fig.add_subplot(grid[1, 0])
        ax.hist(results["ibi"][i], bins=n_bins)
        ax.set_xlabel("ibi")
        ax.set_ylabel("count")
        ax.set_title("inter-burst interval distribution")

        # waveform
        ax = fig.add_subplot(grid[1, 1])
        y = results["waveform_mean"][i]
        y_std = results["waveform_std"][i]
        ax.plot(y, color="black")
        ax.fill_between(x=np.arange(len(y)), y1=y-y_std, y2=y+y_std, alpha=0.5, color="black")
        ax.set_xlabel("time")
        ax.set_ylabel("firing rate")
        ax.set_title("burst waveform")

        plt.tight_layout()

    plt.show()
