import multiprocessing as mp
import numpy as np
from scipy.io import loadmat
import mat73
import os
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from custom_functions import *
import matplotlib.pyplot as plt
from pandas import DataFrame

# function definition
#####################

def organoid_processing(well: int, age: int, spikes_well: np.ndarray, time: np.ndarray, time_ds: np.ndarray,
                        sigma: float, width: float, prominence: float, height: float) -> dict:

    print(f"    ... processing organoid {well}", flush=True)
    results = {"organoid": well, "age": age}

    # extract spikes and calculate firing rate statistics
    spikes = extract_spikes(time, time_ds, spikes_well)
    fr = np.mean(spikes, axis=0)
    results["fr_mean"] = np.mean(fr)
    results["fr_std"] = np.mean(np.std(spikes, axis=0))

    # calculate global dimensionality
    results["dim"] = get_dim(spikes, center=True)

    # calculate global inter-spike interval statistics
    isi = inter_spike_intervals(spikes)
    results["isi_mean"] = np.mean(isi)
    results["isi_std"] = np.std(isi) / np.mean(isi)

    # calculate bursting frequency
    fr_smoothed = gaussian_filter1d(fr, sigma=sigma)
    fr_peaks, peak_props = find_peaks(fr_smoothed, width=width, prominence=prominence, rel_height=height)
    ibi = np.diff(fr_peaks)
    results["ibi_mean"] = np.mean(ibi)
    results["ibi_std"] = np.std(ibi) / np.mean(ibi)

    # calculate intra-burst spiking statistics
    intraburst_results = {"fr_mean": [], "fr_std": [], "isi_mean": [], "isi_std": [], "dim": []}
    for left, right in zip(peak_props["left_ips"], peak_props["right_ips"]):
        spikes_tmp = spikes[:, int(left):int(right)]
        isi_tmp = inter_spike_intervals(spikes_tmp)
        intraburst_results["dim"].append(get_dim(spikes_tmp, center=True))
        intraburst_results["fr_mean"].append(np.mean(spikes_tmp))
        intraburst_results["fr_std"].append(np.mean(np.std(spikes_tmp, axis=0)))
        intraburst_results["isi_mean"].append(np.mean(isi_tmp))
        intraburst_results["isi_std"].append(np.std(isi_tmp) / np.mean(isi_tmp))
    for key, val in intraburst_results.items():
        results[f"intraburst_{key}"] = np.nanmean(val)

    return results

# main script
#############

if __name__ == '__main__':

    mp.set_start_method("spawn")

    # data set specifics
    ####################

    dataset_name = "trujilo_2019"
    path = f"/home/richard/data/{dataset_name}"
    wells = 8
    well_offset = 4
    bursting_sigma = 100
    bursting_width = 1000
    bursting_height = 0.03
    bursting_relheight = 0.98

    # data loading and processing
    #############################

    # prepare results data structure
    results = {
        "organoid": [], "age": [], "dim": [], "isi_mean": [], "isi_std": [], "ibi_mean": [], "ibi_std": [],
        "fr_mean": [], "fr_std": [], "intraburst_dim": [], "intraburst_isi_mean": [], "intraburst_isi_std": [],
        "intraburst_fr_mean": [], "intraburst_fr_std": []
    }

    # prepare age calculation
    year_0 = 16
    month_0 = 7
    day_0 = 1

    for file in os.listdir(path):
        if file.split(".")[-1] == "mat":

            # load data from file
            data = loadmat(f"{path}/{file}", squeeze_me=False)

            # extract data
            lfp = data["LFP"]
            spike_times = data["spikes"]
            time = np.squeeze(data["t_s"])
            time_ds = np.round(np.squeeze(data["t_ds"]), decimals=3) # type: np.ndarray

            # calculate organoid age
            date = file.split(".")[0].split("_")[-1]
            year, month, day = int(date[:2]), int(date[2:4]), int(date[4:])
            month += int((year-year_0)*12)
            day += int((month-month_0)*30)
            age = day - day_0

            print(f"Starting to process organoid data for {date} (age = {age} days)")
            with mp.Pool(processes=wells) as pool:

                # loop over wells/organoids
                res = [pool.apply_async(organoid_processing, (well, age, spike_times[well+well_offset], time, time_ds,
                                                              bursting_sigma, bursting_width, bursting_height,
                                                              bursting_relheight))
                       for well in range(wells)]
                # res = [organoid_processing(well, age, spike_times[well + well_offset], time, time_ds,
                #                            bursting_sigma, bursting_width, bursting_height, bursting_relheight)
                #        for well in range(wells)]

                # save results
                for r in res:
                    res_tmp = r.get()
                    for key in results.keys():
                        results[key].append(res_tmp[key])

                # # plotting: single organoid and time point
                # fig, axes = plt.subplots(nrows=3, figsize=(12, 9))
                # xticks = np.arange(0, stop=time_ds.shape[0], step=50000, dtype=np.int32)
                # fig.suptitle(f"Organoid ID: {well}, Age: {age} days")
                # ax = axes[0]
                # ax.plot(time_ds, fr, label="raw")
                # ax.plot(time_ds, fr_smoothed, label="smoothed")
                # for l, r in zip(peak_props["left_ips"], peak_props["right_ips"]):
                #     ax.axvline(x=time_ds[int(l)], color="black", linestyle="dashed")
                #     ax.axvline(x=time_ds[int(r)], color="red", linestyle="dashed")
                # ax.set_ylabel("rate")
                # ax.set_xlabel("time")
                # ax.legend()
                # ax = axes[1]
                # ax.imshow(spikes, aspect="auto", interpolation="none", cmap="Greys")
                # ax.set_ylabel("neurons")
                # ax.set_xlabel("steps")
                # ax = axes[2]
                # ax.hist(isi)
                # ax.set_xlabel("ISI")
                # ax.set_ylabel("count")
                # plt.tight_layout()
                # plt.show()

            print(f"Finished processing all organoids from {date}.")

    # save results to file
    ######################

    df = DataFrame.from_dict(results)
    df.to_csv(f"{path}/{dataset_name}_summary.csv")
