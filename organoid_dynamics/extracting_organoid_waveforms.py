import multiprocessing as mp
import os
from custom_functions import *
from pandas import DataFrame, MultiIndex, Series
from scipy.io import loadmat

if __name__ == '__main__':

    mp.set_start_method("spawn")

    # data set specifics
    ####################

    # data set specifics
    dataset_name = "trujilo_2019"
    path = f"/home/richard/data/{dataset_name}"
    wells = 8
    well_offset = 4
    exclude = [170224, 170210, 170207, 161216, 161209, 160824, 160803, 160731, 160810, 170303, 161206, 161118]

    # data processing parameters
    tau = 20.0
    sigma = 20.0
    burst_width = 100.0
    burst_sep = 1000.0
    burst_height = 0.5
    burst_relheight = 0.9
    waveform_length = 3000

    # data loading and processing
    #############################

    # prepare results data structure
    results = dict()

    # prepare age calculation
    year_0 = 16
    month_0 = 7
    day_0 = 1

    for file in os.listdir(path):
        if file.split(".")[-1] == "mat":

            if any([str(ID) in file for ID in exclude]):
                continue

            # calculate organoid age
            date = file.split(".")[0].split("_")[-1]
            year, month, day = int(date[:2]), int(date[2:4]), int(date[4:])
            month += int((year - year_0) * 12)
            day += int((month - month_0) * 30)
            age = day - day_0

            # load data from file
            data = loadmat(f"{path}/{file}", squeeze_me=False)

            print(f"Starting to process file {file}")
            with mp.Pool(processes=wells) as pool:

                # loop over wells/organoids
                res = [pool.apply_async(organoid_analysis,
                                        (data, well + well_offset, tau, sigma, burst_width, burst_height,
                                         burst_sep, burst_relheight, waveform_length))
                       for well in range(wells)]

                # save results
                for r in res:
                    res_tmp = r.get()
                    for wave_id, wave in enumerate(res_tmp["waveforms"]):
                        results[(age, res_tmp["organoid"], wave_id)] = wave

            print(f"Finished processing all organoids from file {file}.")

    # save results to file
    ######################

    df = DataFrame(data=np.asarray(list(results.values())).T,
                   columns=MultiIndex.from_tuples(results.keys()),
                   index=np.arange(waveform_length))
    df.to_csv(f"{path}/{dataset_name}_waveforms.csv")
