import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
from scipy.ndimage import gaussian_filter1d
from pandas import DataFrame

# choose condition
drug = "SKF38393"
dose = "Vehicle"
spike_field = "dff_traces_5hz"
speed_field = "speed_traces_5hz"

# meta parameters
speed_threshold = 0.1
acceleration_threshold = 0.1
sigma_speed = 50
sigma_rate = 2

# analysis
##########

data = {"condition": [], "mouse": [], "rate": [], "speed": []}
path = "/run/user/1000/gvfs/smb-share:server=fsmresfiles.fsm.northwestern.edu,share=fsmresfiles/Basic_Sciences/Phys/Kennedylab/Parkerlab/Calcium_v2"
for file in os.listdir(f"{path}/{drug}/{dose}"):

    # load data
    _, mouse_id, *cond = file.split("_")
    condition = "amph" if "amph" in cond else "veh"
    data_tmp = loadmat(f"{path}/{drug}/{dose}/{file}/{condition}_drug.mat", simplify_cells=True)
    spikes = data_tmp[f"{condition}_drug"][spike_field]
    speed = data_tmp[f"{condition}_drug"][speed_field]

    # calculate smooth variables
    smoothed_spikes = np.asarray([gaussian_filter1d(spikes[i, :], sigma=sigma_rate) for i in range(spikes.shape[0])])
    avg_sr = np.mean(smoothed_spikes, axis=0)
    std_sr = np.std(smoothed_spikes, axis=0)
    smoothed_speed = gaussian_filter1d(speed, sigma=sigma_speed)
    smoothed_speed /= np.max(smoothed_speed)

    # calculate acceleration
    speed_diff = np.diff(smoothed_speed, 1)
    speed_diff /= np.max(speed_diff)
    diff2 = np.diff(speed_diff, 1)
    diff2 /= np.max(diff2)

    # plotting
    if condition == "veh":
        fig, axes = plt.subplots(nrows=2, figsize=(12, 6), sharex=True)
        ax = axes[0]
        ax.plot(smoothed_speed, color="black", label="v")
        ax.plot(speed_diff, color="red", label="dv/dt")
        ax.plot(diff2, color="green", label="dv2/d2t")
        ax.legend()
        ax.set_ylabel("speed")
        ax = axes[1]
        ax.plot(avg_sr[:-1] * (smoothed_speed[:-1] < speed_threshold), label="slow")
        ax.plot(avg_sr[:-1] * (smoothed_speed[:-1] >= speed_threshold) * (speed_diff > acceleration_threshold), label="accelerating")
        ax.plot(avg_sr[:-1] * (smoothed_speed[:-1] >= speed_threshold) * (speed_diff < -acceleration_threshold), label="decelerating")
        ax.fill_between(x=np.arange(0, avg_sr.shape[0]), y1=avg_sr - 0.25*std_sr, y2=avg_sr + 0.25*std_sr, alpha=0.5,
                        color="black")
        ax.legend()
        ax.set_ylabel("firing rate")
        ax.set_xlabel("time")
        plt.tight_layout()
        plt.show()

df = DataFrame.from_dict(data)
