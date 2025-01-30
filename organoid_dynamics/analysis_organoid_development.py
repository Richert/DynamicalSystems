import multiprocessing as mp
import numpy as np
from scipy.io import loadmat
import os
from custom_functions import *
import matplotlib.pyplot as plt
from pandas import DataFrame

if __name__ == '__main__':

    mp.set_start_method("spawn")

    # data set specifics
    ####################

    dataset_name = "trujilo_2019"
    path = f"/home/richard/data/{dataset_name}"
    wells = 8
    well_offset = 4
    bursting_sigma = 100
    intraburst_sigma = 10
    bursting_width = 1000
    intraburst_width = 10
    bursting_height = 0.5
    intraburst_height = 0.3
    bursting_relheight = 0.9
    bursting_tau = 10000
    intraburst_tau = 100

    # data loading and processing
    #############################

    # prepare results data structure
    results = {
        "organoid": [], "age": [], "dim": [], "spike_reg": [], "ff": [], "burst_freq": [], "burst_reg": [],
        "rate_avg": [], "rate_het": [], "intraburst_dim": [], "intraburst_spike_reg": [], "intraburst_ff": [],
        "intraburst_rate_avg": [], "intraburst_rate_het": [], "intraburst_freq": []
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
                res = [pool.apply_async(organoid_processing, (well, age, spike_times[well + well_offset], time, time_ds,
                                                              bursting_sigma, intraburst_sigma, bursting_width, intraburst_width,
                                                              bursting_height, intraburst_height, bursting_relheight, bursting_tau,
                                                              intraburst_tau))
                       for well in range(wells)]
                # res = [organoid_processing(well, age, spike_times[well + well_offset], time, time_ds,
                #                            bursting_sigma, intraburst_sigma, bursting_width, intraburst_width,
                #                            bursting_height, intraburst_height, bursting_relheight, bursting_tau,
                #                            intraburst_tau)
                #        for well in range(wells)]

                # save results
                for r in res:
                    res_tmp = r.get()
                    for key in results.keys():
                        results[key].append(res_tmp[key])

            print(f"Finished processing all organoids from {date}.")

    # save results to file
    ######################

    df = DataFrame.from_dict(results)
    df.to_csv(f"{path}/{dataset_name}_summary.csv")
