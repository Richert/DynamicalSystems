import numpy as np
from scipy.io import loadmat
import os
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from pandas import DataFrame

def get_eigs(x: np.ndarray, normalize_variance: bool = False) -> tuple:

    # calculate covariance matrix
    x_centered = np.zeros_like(x)
    for n in range(x.shape[0]):
        x_centered[n, :] = x[n, :] - np.mean(x[n, :])
        if normalize_variance:
            x_centered[n, :] /= (np.std(x_centered[n, :]) + epsilon)
    C = np.cov(x_centered, ddof=0)

    # get eigenvalues and eigenvectors of C
    eigvals, eigvecs = np.linalg.eigh(C)
    eig_idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[eig_idx], eigvecs[:, eig_idx]
    pr = np.sum(eigvals) ** 2 / (np.sum(eigvals ** 2) * len(eigvals))

    return pr, eigvals, eigvecs, C, x_centered

# load processed data or re-process data
path = "/media/storage/DATA/Parkerlab/neural_data" #"/mnt/kennedy_labdata/Parkerlab/neural_data"
save_dir = "/home/richard/data/parker_data"

# plotting parameters
plot_results = False

# choose condition
drugs = ["MP10", "haloperidol", "clozapine", "olanzapine", "xanomeline", "M4PAM", "SCH23390", "SCH39166", "SEP363856", "SKF38393"
         ]
spike_field = "dff_traces_5hz"
speed_field = "speed_traces_5hz"

# define behaviors of interest
behaviors = {
    # "v0": (0.0, 1.0, -0.5, 0.5), "v1": (1.0, 3.0, -0.5, 0.5),
    # "v2": (3.0, 6.0, -0.5, 0.5), "v3": (6.0, np.inf, -0.5, 0.5),
    "p0": (1.0, 3.0, 1, 50), "p1": (3.0, 6.0, 1, 50),
    "p2": (6.0, 10.0, 1, 50), "p3": (10.0, np.inf, 1, 50)
}

# meta parameters
max_neurons = 50
sigma_speed = 1
sigma_rate = 1
epsilon = 1e-15
std_norm = False
window_based = True
gap_window = 5
spn_window = 3

# mouse identity
mice = {"D1":["m085", "m040", "m298", "m404", "f487", "f694", "f857", "f859", "m794", "m797", "m795", "m973",
              "m974", "m659", "m975", "f976", "f977", "f979"],
        "D2": ["m971", "m972", "m106", "m120", "m377", "m380", "f414", "f480", "m483", "m485", "m241", "m242",
               "m523", "f605", "f808", "f811", "f840"]}

# analysis
##########

res = {"drug": [], "condition": [], "mouse": [], "neuron_type": [],
       "mean(r)": [], "D(r)": [], "D(C)": [], "behavior": [], "p(behavior)": []}
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
            v = data_tmp[f"{condition}_drug"][speed_field]
            s = data_tmp[f"{condition}_drug"][spike_field][:max_neurons, :len(v)-1]

            # determine condition
            if dose == "Vehicle" and condition == "veh":
                c = "Control"
            elif dose == "Vehicle" and condition == "amph":
                c = "Amphetamine"
            elif dose == "LowDose" and condition == "amph":
                c = "A + Low Dose"
            elif dose == "HighDose" and condition == "amph":
                c = "A + High Dose"
            else:
                continue

            # calculate smooth variables
            s2 = np.asarray([gaussian_filter1d(s[i, :], sigma=sigma_rate) for i in range(s.shape[0])])
            v2 = gaussian_filter1d(v, sigma=sigma_speed)
            a2 = np.diff(v2)
            v2 = v2[:-1]

            # calculate result variables and save data
            for b, thresholds in behaviors.items():

                # get index where mouse shows target behavior
                if "v" in b:
                    idx = (v2 >= thresholds[0]) & (v2 < thresholds[1]) & (a2 >= thresholds[2]) & (a2 < thresholds[3])
                else:
                    idx2, props = find_peaks(v2, distance=gap_window, prominence=(thresholds[0], thresholds[1]),
                                             width=(thresholds[2], thresholds[3]), plateau_size=(0, 2), rel_height=1.0)
                    idx = np.zeros_like(v2) > 0.0
                    for l, r in zip(props["left_ips"], props["right_ips"]):
                        idx[int(np.round(l, decimals=0)):int(np.round(r, decimals=0))] = True
                idx[0] = False
                if np.sum(idx) < spn_window:
                    continue

                if window_based:

                    idx_diff = np.diff(1.0 * idx)
                    starts, stops = np.argwhere(idx_diff > 0.0).squeeze(), np.argwhere(idx_diff < 0.0).squeeze()
                    behavior_data = {"r": [], "D(C)": [], "D(r)": [], "r_dist": []}
                    try:
                        for start, stop in zip(starts, stops):

                            if stop - start < spn_window:
                                continue

                            # get covariance matrix and calculate dimensionality
                            s2_idx = s2[:, start:stop]
                            pr2, eigvals2, eigvecs2, C2, s_c2 = get_eigs(s2_idx, normalize_variance=std_norm)
                            pcs2 = eigvecs2.T @ s2_idx
                            if not np.isfinite(pr2):
                                continue

                            # save window data
                            r2 = np.mean(s2_idx, axis=1)
                            behavior_data["r"].append(np.mean(r2))
                            behavior_data["D(C)"].append(pr2)
                            behavior_data["D(r)"].append(np.sum(r2) ** 2 / (np.sum(r2 ** 2) * len(r2)))
                            behavior_data["r_dist"].append(r2)

                        # store results
                        pr = np.mean(behavior_data["D(C)"])
                        if np.isfinite(pr):
                            res["drug"].append(drug)
                            res["condition"].append(c)
                            res["mouse"].append(mouse_id)
                            res["neuron_type"].append("D1" if mouse_id in mice["D1"] else "D2")
                            res["mean(r)"].append(np.mean(behavior_data["r"]))
                            res["D(r)"].append(np.mean(behavior_data["D(r)"]))
                            res["D(C)"].append(pr)
                            res["behavior"].append(b)
                            res["p(behavior)"].append(np.mean(idx))

                    except TypeError:
                        pass

                else:

                    # get covariance matrix and calculate dimensionality
                    s2_idx = s2[:, idx]
                    pr2, eigvals2, eigvecs2, C2, s_c2 = get_eigs(s2_idx, normalize_variance=std_norm)
                    pcs2 = eigvecs2.T @ s2_idx
                    if not np.isfinite(pr2):
                        continue

                    # store results
                    r = np.mean(s2_idx, axis=1)
                    if np.isfinite(pr2):
                        res["drug"].append(drug)
                        res["condition"].append(c)
                        res["mouse"].append(mouse_id)
                        res["neuron_type"].append("D1" if mouse_id in mice["D1"] else "D2")
                        res["mean(r)"].append(np.mean(r))
                        res["D(r)"].append(np.sum(r)**2 / (np.sum(r**2) * len(r)))
                        res["D(C)"].append(pr2)
                        res["behavior"].append(b)
                        res["p(behavior)"].append(np.mean(idx))

    print(f"Finished processing data for drug = {drug}.")

# save data
df = DataFrame.from_dict(res)
df.to_csv(f"{save_dir}/spn_behavior_p{'_normalized' if std_norm else ''}{'_multi_window' if window_based else '_single_window'}.csv")

# plotting
if plot_results:

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("TkAgg")
    matplotlib.rcParams["font.size"] = 14
    matplotlib.rcParams["axes.labelsize"] = 14
    matplotlib.rcParams["lines.markersize"] = 12.0
    matplotlib.rcParams["lines.linewidth"] = 2.0
    import seaborn as sb

    sb.set_palette("colorblind")
    df.sort_values(["condition", "neuron_type"], inplace=True)
    for key in ["D(C)", "D(r)"]:
        for drug in drugs:
            fig, ax = plt.subplots(figsize=(10, 6))
            sb.lineplot(df.loc[df.loc[:, "drug"] == drug, :],
                        x="behavior", y=key, hue="condition", style="neuron_type", ax=ax,
                        markers=True, dashes=True, err_style='bars', errorbar="se")
            ax.set_title(drug)
            plt.tight_layout()
            fig.canvas.draw()
    plt.show()
