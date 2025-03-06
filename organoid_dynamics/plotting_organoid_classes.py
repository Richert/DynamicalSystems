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
