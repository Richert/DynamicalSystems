import multiprocessing as mp
import numpy as np
from scipy.io import loadmat
import os
from custom_functions import *
import matplotlib.pyplot as plt
from pandas import DataFrame


def organoid_analysis(well: int, age: int, spikes_well: np.ndarray, lfps_well: np.ndarray, time: np.ndarray,
                      time_ds: np.ndarray, sigma: float, width: float, width2: float, prominence: float,
                      prominence2: float, height: float, tau: int, nperseg: int, fmax: float) -> dict:

    results = organoid_spike_analysis(well, age, spikes_well, time, time_ds, sigma, width, width2,
                                      prominence, prominence2, height, tau, return_spikes=False)
    results_lfp = organoid_lfp_analysis(well, age, lfps_well, time_ds, nperseg, fmax)
    results["lfp_dim"] = results_lfp["dim"]
    results["lfp_var"] = results_lfp["lfp_var"]
    results["max_freq"] = results_lfp["max_freq"]
    results["max_pow"] = results_lfp["max_pow"]

    return results


if __name__ == '__main__':

    mp.set_start_method("spawn")

    # data set specifics
    ####################

    dataset_name = "trujilo_2019"
    path = f"/home/richard-gast/Documents/data/{dataset_name}"
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
    burst_freq_threshold = 0.005

    exclude = [170224, 170210, 170207, 161216, 161209, 160824, 160803, 160731, 160810, 170303, 161206, 161118]

    # data loading and processing
    #############################

    # prepare results data structure
    results = {
        "organoid": [], "age": [], "dim": [], "spike_reg": [], "burst_freq": [], "burst_reg": [],
        "rate_avg": [], "rate_het": [], "intraburst_dim": [], "intraburst_spike_reg": [],
        "intraburst_rate_avg": [], "intraburst_rate_het": [], "intraburst_freq": [],
        "lfp_dim": [], "lfp_var": [], "max_freq": [], "max_pow": []
    }

    # prepare age calculation
    year_0 = 16
    month_0 = 7
    day_0 = 1

    for file in os.listdir(path):
        if file.split(".")[-1] == "mat":

            if any([str(ID) in file for ID in exclude]):
                continue

            # load data from file
            data = loadmat(f"{path}/{file}", squeeze_me=False)

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

            print(f"Starting to process organoid data for {date} (age = {age} days)")
            with mp.Pool(processes=wells) as pool:

                # loop over wells/organoids
                res = [pool.apply_async(organoid_analysis,
                                        (well, age, spike_times[well + well_offset], lfp[well + well_offset], time, time_ds,
                                         sigma, bursting_width, intraburst_width, bursting_height, intraburst_height,
                                         bursting_relheight, tau, lfp_nperseg, lfp_fmax))
                       for well in range(wells)]

                # save results
                for r in res:
                    res_tmp = r.get()
                    if res_tmp["burst_freq"] > burst_freq_threshold:
                        for key in results.keys():
                            results[key].append(res_tmp[key])

            print(f"Finished processing all organoids from {date}.")

    # save results to file
    ######################

    df = DataFrame.from_dict(results)
    df.to_csv(f"{path}/{dataset_name}_summary.csv")
