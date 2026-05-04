import numpy as np
from scipy.io import loadmat
import os
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths, correlate
from pandas import DataFrame
import pickle

def get_cov(x: np.ndarray, normalize_variance: bool = False) -> tuple:

    x_centered = np.zeros_like(x)
    for n in range(x.shape[0]):
        x_centered[n, :] = x[n, :] - np.mean(x[n, :])
        if normalize_variance:
            x_centered[n, :] /= (np.std(x_centered[n, :]) + epsilon)
    C = np.cov(x_centered, ddof=0)

    return C, x_centered

def get_eigs(C: np.ndarray) -> tuple:

    # get eigenvalues and eigenvectors of C
    eigvals, eigvecs = np.linalg.eigh(C)
    eig_idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[eig_idx], eigvecs[:, eig_idx]
    pr = np.sum(eigvals) ** 2 / (np.sum(eigvals ** 2) * len(eigvals))

    return pr, eigvals, eigvecs

# load processed data or re-process data
path = "/home/rgast/data/parker_data/neural_data" #"/mnt/kennedy_labdata/Parkerlab/neural_data"
save_dir = "/home/rgast/data/parker_data"

# plotting parameters
plot_results = True

# choose condition
drugs = [
         "haloperidol", "xanomeline", "MP10",
         #"MP10", "clozapine", "olanzapine", "xanomeline", "M4PAM",
         #"SCH23390", "SCH39166", "SEP363856", "SKF38393"
         ]
spike_field = "dff_traces_5hz"
speed_field = "speed_traces_5hz"

# define behaviors of interest
behaviors = {
    # "v0": (0.0, 1.0, -0.5, 0.5), "v1": (1.0, 3.0, -0.5, 0.5),
    # "v2": (3.0, 6.0, -0.5, 0.5), "v3": (6.0, np.inf, -0.5, 0.5),
    "p3": (9.0, 30.0, 12, 30), "p2": (9.0, 30.0, 1, 12),
    "p1": (3.0, 9.0, 10, 20), "p0": (3.0, 9.0, 1, 10),
}

# meta parameters
max_neurons = 50
sigma_speed = 1
sigma_rate = 1
epsilon = 1e-15
std_norm = False
gap_window = 5
corr_window = 50
spn_window = 5
spike_scaling = 20.0 * 60.0

# mouse identity
mice = {"D1":["m085", "m040", "m298", "m404", "f487", "f694", "f857", "f859", "m794", "m797", "m795", "m973",
              "m974", "m659", "m975", "f976", "f977", "f979"],
        "D2": ["m971", "m972", "m106", "m120", "m377", "m380", "f414", "f480", "m483", "m485", "m241", "m242",
               "m523", "f605", "f808", "f811", "f840"]}

# analysis
##########

res = {"drug": [], "condition": [], "mouse": [], "SPNs": [], "behavior": [],
       "mean(r)": [], "D(r)": [], "D(C)": [], "p(behavior)": [], "C": [], "r": [], "AC(v)": []}
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
            s = data_tmp[f"{condition}_drug"][spike_field][:max_neurons, :len(v)-1] * spike_scaling
            if s.shape[0] < max_neurons:
                continue

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

            # normalize velocity
            v_scaled = v2 - np.mean(v2)
            v_scaled /= np.std(v_scaled)

            # calculate result variables and save data
            b_indices = np.zeros_like(v2)
            for b, thresholds in behaviors.items():

                # get index where mouse shows target behavior
                if "v" in b:
                    idx = (v2 >= thresholds[0]) & (v2 < thresholds[1]) & (a2 >= thresholds[2]) & (a2 < thresholds[3])
                else:
                    idx2, props = find_peaks(v2, distance=gap_window, prominence=(thresholds[0], thresholds[1]),
                                             width=(thresholds[2], thresholds[3]), plateau_size=(0, 3), rel_height=0.6)
                    idx = np.zeros_like(v2) > 0.0
                    width_data = peak_widths(v2, idx2, rel_height=0.8,
                                             prominence_data=(props["prominences"], props["left_bases"],
                                                              props["right_bases"])
                                             )
                    for l, r in zip(width_data[2], width_data[3]):
                        idx[int(np.round(l, decimals=0)):int(np.round(r, decimals=0))] = True
                idx[0] = False
                idx[b_indices > 0.0] = False
                b_indices[idx == True] = 1.0

                idx_diff = np.diff(1.0 * idx)
                starts, stops = np.argwhere(idx_diff > 0.0).squeeze(axis=1), np.argwhere(idx_diff < 0.0).squeeze(axis=1)
                behavior_data = {"r": [], "C": [], "v": []}
                if len(starts) < 1 or len(stops) < 1:
                    continue
                for start, stop in zip(starts, stops):

                    if stop - start < spn_window:
                        continue

                    # get covariance matrix and calculate dimensionality
                    s2_idx = s2[:, start:stop]
                    C2, _ = get_cov(s2_idx, normalize_variance=std_norm)

                    # calculate velocity autocorrelation
                    center = int((start + stop) / 2)
                    wh = int(corr_window / 2)
                    if (center - wh) >= 0 and (center + wh) < len(v_scaled):
                        v_tmp = v_scaled[center - wh:center + wh]
                        v_c = correlate(v_tmp, v_tmp, mode="full")
                        behavior_data["v"].append(v_c)

                    # save window data
                    r2 = np.mean(s2_idx, axis=1)
                    behavior_data["r"].append(r2)
                    behavior_data["C"].append(C2)

                # store results
                C = np.mean(behavior_data["C"], axis=0)
                r = np.mean(behavior_data["r"], axis=0)
                vc = np.mean(behavior_data["v"], axis=0)
                pr, eigvals, eigvecs = get_eigs(C)
                if np.isfinite(pr):
                    res["drug"].append(drug)
                    res["condition"].append(c)
                    res["mouse"].append(mouse_id)
                    res["SPNs"].append("D1" if mouse_id in mice["D1"] else "D2")
                    res["mean(r)"].append(np.mean(behavior_data["r"]))
                    res["D(r)"].append(np.sum(r)**2/(np.sum(r**2)*len(r)))
                    res["D(C)"].append(pr)
                    res["behavior"].append(b)
                    res["p(behavior)"].append(np.mean(idx))
                    res["C"].append(C)
                    res["r"].append(r)
                    res["AC(v)"].append(vc)

    print(f"Finished processing data for drug = {drug}.")

# save data
Cs, rs, ACs = np.asarray(res.pop("C")), np.asarray(res.pop("r")), np.asarray(res.pop("AC(v)"))
res["Delta_C"], res["Delta_r"] = np.zeros((len(Cs),)), np.zeros((len(Cs),))
mice, mice_indices = np.unique(res["mouse"], return_inverse=True)
for i in range(len(mice)):
    idx1 = mice_indices == i
    for p in behaviors.keys():
        idx2 = np.asarray(res["behavior"]) == p
        idx3 = np.asarray(res["condition"]) == "Control"
        idx = idx1 & idx2 & idx3
        try:
            C0, r0 = Cs[idx][0], rs[idx][0]
            for c in ["Amphetamine", "A + Low Dose", "A + High Dose"]:
                    idx3 = np.asarray(res["condition"]) == c
                    idx = idx1 & idx2 & idx3
                    C1, r1 = Cs[idx][0], rs[idx][0]
                    Delta_C = np.corrcoef(C0.flatten(), C1.flatten())[0, 1]
                    Delta_r = np.corrcoef(r0, r1)[0, 1]
                    res["Delta_C"][idx] = Delta_C
                    res["Delta_r"][idx] = Delta_r
        except (IndexError, ValueError):
            pass

# store 1-dim data
res["Delta_C"], res["Delta_r"] = res["Delta_C"].tolist(), res["Delta_r"].tolist()
df = DataFrame.from_dict(res)
df.to_csv(f"{save_dir}/spn_behavior_{'v' if 'v' in df.at[0, 'behavior'] else 'p'}.csv")

# store multi-dim data
pickle.dump({"drug": np.asarray(res["drug"]), "condition": np.asarray(res["condition"]),
             "mouse": np.asarray(res["mouse"]), "SPNs": np.asarray(res["SPNs"]),
             "behavior": np.asarray(res["behavior"]), "r": rs, "AC(v)": ACs, "C": Cs},
            open(f"{save_dir}/spn_behavior_{'v' if 'v' in df.at[0, 'behavior'] else 'p'}.pkl", "wb"))

# plotting
if plot_results:

    import matplotlib.pyplot as plt
    import matplotlib
    import seaborn as sb
    matplotlib.use("TkAgg")
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams["lines.markersize"] = 12.0
    plt.rcParams["lines.linewidth"] = 2.0

    sb.set_palette("colorblind")
    df.sort_values(["condition", "SPNs", "behavior"], inplace=True)
    for key in ["D(C)", "D(r)"]:
        for drug in drugs:
            fig, ax = plt.subplots(figsize=(10, 6))
            sb.lineplot(df.loc[df.loc[:, "drug"] == drug, :],
                        x="behavior", y=key, hue="condition", style="SPNs", ax=ax,
                        markers=True, dashes=True, err_style='bars', errorbar="se")
            ax.set_title(drug)
            plt.tight_layout()
            fig.canvas.draw()
    plt.show()
