import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import seaborn as sns
from pandas import DataFrame
from scipy.ndimage import gaussian_filter1d

# choose condition
drug = "SKF38393"
dose = "Vehicle"
spike_field = "dff_traces_5hz"
speed_field = "speed_traces_5hz"

# meta parameters
sigma_speed = 20
sigma_rate = 20

# analysis
##########

data = {"condition": [], "mouse": [], "rate": [], "speed": []}
path = "/mnt/kennedy_lab_data/Parkerlab/Calcium_v2"
for file in os.listdir(f"{path}/{drug}/{dose}"):

    # load data
    _, mouse_id, *cond = file.split("_")
    condition = "amph" if "amph" in cond else "veh"
    data_tmp = loadmat(f"{path}/{drug}/{dose}/{file}/{condition}_drug.mat", simplify_cells=True)
    rate = np.mean(data_tmp[f"{condition}_drug"][spike_field], axis=0)
    speed = data_tmp[f"{condition}_drug"][speed_field]

    # smooth data
    rate = gaussian_filter1d(rate, sigma=sigma_rate)
    speed = gaussian_filter1d(speed, sigma=sigma_speed)

    # save data
    n = rate.shape[0]
    data["condition"].extend([condition] * n)
    data["mouse"].extend([mouse_id] * n)
    data["rate"].extend(rate.squeeze().tolist())
    data["speed"].extend(speed.squeeze().tolist())
    print(f"File {file} finished.")

df = DataFrame.from_dict(data)

# plotting
##########

g = sns.lmplot(data=df.loc[df["condition"] == "veh", :], x="speed", y="rate", hue="mouse", height=6, aspect=2)
g.ax.set_title(f"condition: vehicle")
plt.tight_layout()

g = sns.lmplot(data=df.loc[df["condition"] == "amph", :], x="speed", y="rate", hue="mouse", height=6, aspect=2)
g.ax.set_title(f"condition: amphetamine")
plt.tight_layout()

plt.show()
