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
from pandas import DataFrame, read_csv
import pickle

def get_ff(rates: np.ndarray) -> np.ndarray:
    n = rates.shape[1]
    ff = np.zeros((n,))
    for i in range(n):
        ff[i] = np.var(rates[:, i]) / (np.mean(rates[:, i]) + epsilon)
    return ff

# load processed data or re-process data
path = f"/mnt/kennedy_lab_data/Parkerlab/neural_data"
process_data = True

# choose condition
drugs = ["clozapine", "olanzapine", "xanomeline", "MP10", "haloperidol", "M4PAM", "SCH23390", "SCH39166", "SEP363856", "SKF38393",
         ]
spike_field = "dff_traces_5hz"
speed_field = "speed_traces_5hz"
d12_combined = True

# meta parameters
sigma_speed = 5
sigma_rate = 5
bv_bins = 5
max_speed = 10.0
bins = np.round(np.linspace(0.0, 1.0, num=bv_bins+1)*max_speed, decimals=1)
epsilon = 1e-12
gap_window = 1

# mouse identity
mice = {"D1":["m085", "m040", "m298", "m404", "f487", "f694", "f857", "f859", "m794", "m797", "m795"],
        "D2": ["m971", "m972", "m106", "m120", "m377", "m380", "f414", "f480", "m483", "m485"]}

# analysis
##########

if process_data:
    res = {"drug": [], "dose": [], "condition": [], "mouse": [], "neuron_type": [],
            "mean rate": [], "rate fluctuation": [], "dimensionality 1": [], "dimensionality 2": [],
            "v": [], "p(v)": [], "fano factor": []}
    signals = {"drug": [], "dose": [], "condition": [], "mouse": [], "neuron_type": [], "C": [], "pc_rates": []}
    results = {"results": res.copy(), "data": signals} #"speed_d1": data.copy(), "speed_d2": data
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

                # calculate smooth variables
                smoothed_spikes = np.asarray([gaussian_filter1d(spikes[i, :], sigma=sigma_rate) for i in range(spikes.shape[0])])
                smoothed_speed = gaussian_filter1d(speed, sigma=sigma_speed)

                # get speed histogram
                speed_hist = np.histogram(smoothed_speed, bins=bins)[0]
                speed_hist = speed_hist / np.sum(speed_hist)

                # center firing rates
                rates_centered = np.zeros_like(smoothed_spikes)
                for i in range(smoothed_spikes.shape[0]):
                    rates_centered[i, :] = smoothed_spikes[i, :] - np.mean(smoothed_spikes[i, :])
                    rates_centered[i, :] /= (np.std(rates_centered[i, :]) + epsilon)

                # calculate participation ratio and save data
                bv = smoothed_speed
                for bin in range(bv_bins):

                    rates = smoothed_spikes[:, :len(bv)]
                    rates_c = rates_centered[:, :len(bv)]
                    idx = (bins[bin] <= bv) & (bv < bins[bin+1])
                    if len(idx) < rates.shape[0]:
                        print(f"No data for velocity bin {bin} of file {file}.")
                        continue

                    # calculate rate statistics
                    population_rate = np.mean(rates[:, idx], axis=0)
                    population_variability = np.std(rates[:, idx], axis=0)

                    # SVD analysis
                    U, s, V = np.linalg.svd(rates_c[:, idx], full_matrices=False)
                    pr = np.sum(s)**2/(np.sum(s**2)*rates_c.shape[0])
                    if not np.isfinite(pr):
                        continue

                    # SVD analysis 2 and fano factor analysis
                    try:
                        v_indices = np.argwhere(idx == True).squeeze()
                        idx_diff = np.diff(v_indices)
                        PR2, FFs = [], []
                        i = 0
                        for gap_idx in np.argwhere(idx_diff > gap_window).squeeze():
                            rate_chunk = rates_c[:, v_indices[i:gap_idx+1]]
                            U2, s2, V2 = np.linalg.svd(rate_chunk, full_matrices=False)
                            pr2 = np.sum(s2) ** 2 / (np.sum(s2 ** 2) * (gap_idx + 1 - i))
                            if np.isfinite(pr2):
                                PR2.append(pr2)
                            FFs.append(get_ff(rate_chunk))
                            i += gap_idx + 1
                        pr2 = np.mean(PR2)
                        ff = np.mean(FFs)
                    except (TypeError, ValueError):
                        pr2 = pr
                        ff = 0.0

                    # store results
                    res, signals = results["results"], results["data"]
                    res["drug"].append(drug)
                    res["dose"].append(dose)
                    res["condition"].append(condition)
                    res["mouse"].append(mouse_id)
                    res["neuron_type"].append("D1" if mouse_id in mice["D1"] else "D2")
                    res["mean rate"].append(np.mean(population_rate))
                    res["rate fluctuation"].append(np.std(population_rate))
                    res["dimensionality 1"].append(pr)
                    res["dimensionality 2"].append(pr2)
                    res["v"].append(np.round((bins[bin] + bins[bin + 1]) / 2, decimals=1))
                    res["p(v)"].append(speed_hist[bin])
                    res["fano factor"].append(ff)
                    signals["drug"].append(drug)
                    signals["dose"].append(dose)
                    signals["condition"].append(condition)
                    signals["mouse"].append(mouse_id)
                    signals["neuron_type"].append("D1" if mouse_id in mice["D1"] else "D2")
                    signals["C"].append(rates_c[:, idx] @ rates_c[:, idx].T)
                    signals["pc_rates"].append((U.T @ rates_c[:, idx])[:3, :])

        print(f"Finished processing data for drug = {drug}.")

    # save data
    df = DataFrame.from_dict(results["results"])
    df.to_csv(f"/home/rgast/data/parker_data/spn_dimensionality_data.csv")
    pickle.dump(results["data"], open("/home/rgast/data/parker_data/spn_dimensionality_data.pkl", "wb"))

else:

    df = read_csv(f"/home/rgast/data/parker_data/spn_dimensionality_data.csv")

# plotting
sb.set_palette("colorblind")
df.sort_values(["condition", "dose"], inplace=True)
for key in ["fano factor"]:
    for drug in drugs:
        if d12_combined:
            fig, ax = plt.subplots(figsize=(10, 6))
            sb.lineplot(df.loc[df.loc[:, "drug"] == drug, :],
                        x="v", y=key, hue="dose", style="condition", ax=ax,
                        markers=True, dashes=True, err_style='bars', errorbar="se")
            ax.set_title(drug)
            plt.tight_layout()
            fig.canvas.draw()
            fig.savefig(f"/home/rgast/data/parker_data/{drug}_{key}_d1_d2_combined.svg")
        else:
            fig, axes = plt.subplots(ncols=2, figsize=(12, 6), sharex=True, sharey=True)
            fig.suptitle(drug)
            for i, d in enumerate(["D1", "D2"]):
                ax = axes[i]
                sb.lineplot(df.loc[(df.loc[:, "drug"] == drug) & (df.loc[:, "neuron_type"] == d), :],
                            x="v", y=key, hue="dose", style="condition", ax=ax,
                            markers=True, dashes=True, err_style='bars', errorbar="se")
                ax.set_title(f"{d}")
            plt.tight_layout()
            fig.canvas.draw()
            fig.savefig(f"/home/rgast/data/parker_data/{drug}_{key}_d1_d2_split.svg")
plt.show()
