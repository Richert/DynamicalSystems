import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["axes.labelsize"] = 14
matplotlib.rcParams["lines.markersize"] = 12.0
matplotlib.rcParams["lines.linewidth"] = 2.0
import seaborn as sb
from scipy.io import loadmat
import os
from scipy.ndimage import gaussian_filter1d
from pandas import DataFrame

# choose condition
drugs = [
    "MP10", "haloperidol"
         ]
spike_field = "dff_traces_5hz"
speed_field = "speed_traces_5hz"
d12_combined = True

# meta parameters
sigma_speed = 5
sigma_rate = 5
bv_bins = 10
max_speed = 10.0
bins = np.round(np.linspace(0.0, 1.0, num=bv_bins+1)*max_speed, decimals=1)

# mouse identity
mice = {"D1":["m085", "m040", "m298", "m404", "f487", "f694", "f857", "f859", "m794", "m797", "m795"],
        "D2": ["m971", "m972", "m106", "m120", "m377", "m380", "f414", "f480", "m483", "m485"]}

# analysis
##########

data = {"drug": [], "dose": [], "condition": [], "mouse": [], "neuron_type": [], "v": [], "p(v)": []}
results = {"speed": data.copy()} #"speed_d1": data.copy(), "speed_d2": data
path = f"/mnt/kennedy_lab_data/Parkerlab/neural_data"
for drug in drugs:

    print(f"Starting to process data for drug = {drug}.")

    for dose in ["Vehicle", "LowDose", "HighDose"]:
        for file in os.listdir(f"{path}/{drug}/{dose}"):

            # load data
            try:
                _, mouse_id, *cond = file.split("_")
                condition = "amph" if "amph" in cond else "veh"
                data_tmp = loadmat(f"{path}/{drug}/{dose}/{file}/{condition}_drug.mat", simplify_cells=True)
            except NotADirectoryError:
                continue
            spikes = data_tmp[f"{condition}_drug"][spike_field]
            speed = data_tmp[f"{condition}_drug"][speed_field]

            # calculate smooth speed
            smoothed_speed = gaussian_filter1d(speed, sigma=sigma_speed)

            # get speed histogram
            speed_hist = np.histogram(smoothed_speed, bins=bins)[0]
            speed_hist = speed_hist / np.sum(speed_hist)

            # save data per velocity bin
            for key, bv in zip(list(results.keys()), [smoothed_speed]):
                for bin in range(bv_bins):

                    # store results
                    data = results[key]
                    data["drug"].append(drug)
                    data["dose"].append(dose)
                    data["condition"].append("amphetamine" if condition == "amph" else "control")
                    data["mouse"].append(mouse_id)
                    data["neuron_type"].append("D1" if mouse_id in mice["D1"] else "D2")
                    data["v"].append(np.round((bins[bin] + bins[bin+1])/2, decimals=1))
                    data["p(v)"].append(speed_hist[bin])

    print(f"Finished processing data for drug = {drug}.")

# plotting
sb.set_palette("colorblind")
for bv_key in results.keys():
    df = DataFrame.from_dict(results[bv_key])
    df.sort_values(["condition", "dose"], inplace=True)
    for drug in drugs:
        fig, axes = plt.subplots(ncols=3, figsize=(12, 6))
        for i, dose in enumerate(["Vehicle", "LowDose", "HighDose"]):
            ax = axes[i]
            sb.barplot(df.loc[(df.loc[:, "drug"] == drug) & (df.loc[:, "dose"] == dose), :],
                        x="v", y="p(v)", hue="condition", ax=ax, errorbar="se")
            ax.set_title(dose)
            ax.set_xticks([np.round(x, decimals=1) for x in ax.get_xticks()[::2]])
            ax.set_ylim([0.0, 1.0])
        fig.suptitle(drug)
        plt.tight_layout()
        fig.canvas.draw()
        fig.savefig(f"/home/rgast/data/parker_data/{drug}_velocities.svg")
plt.show()
