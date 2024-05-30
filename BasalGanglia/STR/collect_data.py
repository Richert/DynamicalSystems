import numpy as np
from scipy.io import loadmat
from scipy.signal import decimate
import os
from pandas import DataFrame

# choose condition
drug = "SKF38393"
dose = "Vehicle"
spike_field = "events_5hz"
speed_field = "speed_traces_5hz"

# meta parameters
sigma = 2
sr = 5

# analysis
##########

data = {"condition": [], "mouse": [], "rate_mean": [], "rate_std": [],
        "velocity": [], "angle": [], "time": []}

path = "/run/user/1000/gvfs/smb-share:server=fsmresfiles.fsm.northwestern.edu,share=fsmresfiles/Basic_Sciences/Phys/Kennedylab/Parkerlab/Calcium_v2"
path2 = "/run/user/1000/gvfs/smb-share:server=fsmresfiles.fsm.northwestern.edu,share=fsmresfiles/Basic_Sciences/Phys/Kennedylab/Parkerlab/Behavior"

# collect neural and velocity data
for file in os.listdir(f"{path}/{drug}/{dose}"):

    # load data
    _, mouse_id, *cond = file.split("_")
    condition = "amph" if "amph" in cond else "veh"
    data_tmp = loadmat(f"{path}/{drug}/{dose}/{file}/{condition}_drug.mat", simplify_cells=True)
    spikes = data_tmp[f"{condition}_drug"][spike_field]
    speed = data_tmp[f"{condition}_drug"][speed_field]

    # calculate smooth variables
    avg_sr = np.mean(spikes, axis=0)
    std_sr = np.std(spikes, axis=0)

    # save data
    n = len(avg_sr)
    data["condition"].extend([condition] * n)
    data["mouse"].extend([mouse_id] * n)
    data["time"].extend(np.arange(0, n))
    data["rate_mean"].extend(avg_sr[:n].tolist())
    data["rate_std"].extend(std_sr[:n].tolist())
    data["velocity"].extend(speed[:n].tolist())
    data["angle"].extend(np.zeros((n,)).tolist())

df = DataFrame.from_dict(data)

# collect tracking data
for cond in ["Amph", "Control"]:
    for file in os.listdir(f"{path2}/{drug}/{dose}/{cond}/output_v1_8"):

        # load data
        _, mouse_id, *_ = file.split("_")
        condition = "amph" if cond == "Amph" else "veh"
        data_tmp = loadmat(f"{path2}/{drug}/{dose}/{cond}/output_v1_8/{file}/{file}_custom_feat_top_v1_8.mat",
                           simplify_cells=True)

        # extract and downsample angle
        idx = np.argwhere(["top_m0_angle_nose_neck_tail" in f for f in data_tmp["features"]]).squeeze()
        angle = data_tmp["data_smooth"][:, idx]
        angle = decimate(angle, int(data_tmp["fps"]/sr))

        # add angle to data
        df_tmp = df.loc[(df["mouse"] == mouse_id) & (df["condition"] == condition), :]
        df_tmp = df_tmp.sort_values("time", ascending=True, axis=0)
        start = angle.shape[0] - df_tmp.shape[0]
        df_tmp.loc[:, "angle"] = angle[start:]
        df.loc[(df["mouse"] == mouse_id) & (df["condition"] == condition), :] = df_tmp

path = "/run/user/1000/gvfs/smb-share:server=fsmresfiles.fsm.northwestern.edu,share=fsmresfiles/Basic_Sciences/Phys/Kennedylab/Parkerlab/Richard/data"
df.to_csv(f"{path}/{drug}_{dose}_combined.csv")
