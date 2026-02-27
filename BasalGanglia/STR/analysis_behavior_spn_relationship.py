import numpy as np
from scipy.io import loadmat
import os
from scipy.ndimage import gaussian_filter1d
from pandas import DataFrame, read_csv
import pickle

# load processed data or re-process data
path = "/mnt/kennedy_labdata/Parkerlab/neural_data"
save_dir = "/home/rgast/data/parker_data"
process_data = False
plot_results = True

# choose condition
drugs = ["clozapine", "olanzapine", "xanomeline", "MP10", "haloperidol", "M4PAM",
         "SCH23390", "SCH39166", "SEP363856", "SKF38393"]
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
gap_window = 5

# mouse identity
mice = {"D1":["m085", "m040", "m298", "m404", "f487", "f694", "f857", "f859", "m794", "m797", "m795", "m973",
              "m974", "m659", "m975", "f976", "f977", "f979"],
        "D2": ["m971", "m972", "m106", "m120", "m377", "m380", "f414", "f480", "m483", "m485", "m241", "m242",
               "m523", "f605", "f808", "f811", "f840"]}

# analysis
##########

if process_data:
    res = {"drug": [], "dose": [], "condition": [], "mouse": [], "neuron_type": [],
            "mean(r|t)": [], "std(r|t)": [], "std(r|i)": [],"dimensionality": [],
            "v": [], "p(v)": [], "corr(v,pc1)": [], "corr(v,pc2)": []}
    signals = {"drug": [], "dose": [], "condition": [], "mouse": [], "neuron_type": [], "v": [],
               "C": [], "pc1": [], "pc2": [], "v(t)": []}
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

                # calculate result variables and save data
                bv = smoothed_speed
                for bin in range(bv_bins):

                    # bin-based analyses
                    try:

                        # find time windows where mouse runs at particular velocity
                        idx = (bins[bin] <= bv) & (bv < bins[bin + 1])
                        v_indices = np.argwhere(idx == True).squeeze()
                        idx_diff = np.diff(v_indices)
                        rates = smoothed_spikes[:, :len(bv)]

                        # go over different time bins
                        PR, CC1, CC2, PC1, PC2, PC_n, V, Cs, Rt, Rn = [], [], [], [], [], [], [], [], [], []
                        i = 0
                        for gap_idx in np.argwhere(idx_diff > gap_window).squeeze():

                            # find time window of consistent velocity
                            idx2 = v_indices[i:gap_idx+1]
                            if len(idx2) < gap_window:
                                continue

                            # center firing rates
                            rates = smoothed_spikes[:, :len(bv)]
                            rate_chunk = rates[:, idx2]
                            rates_c = np.zeros_like(rate_chunk)
                            for j in range(rate_chunk.shape[0]):
                                rates_c[j, :] = rate_chunk[j, :] - np.mean(rate_chunk[j, :])
                                rates_c[j, :] /= (np.std(rate_chunk[j, :]) + epsilon)

                            # get covariance matrix and calculate dimensionality
                            C = np.cov(rates_c)
                            eigvals, eigvecs = np.linalg.eigh(C)
                            eig_idx = np.argsort(eigvals)[::-1]
                            pr2 = np.sum(eigvals) ** 2 / (np.sum(eigvals ** 2) * (gap_idx + 1 - i))
                            if not np.isfinite(pr2):
                                continue
                            Cs.append(C)
                            PR.append(pr2)

                            # get correlation between PCs and velocity
                            pcs = eigvecs[:, eig_idx].T @ rates_c
                            CC1.append(np.corrcoef(smoothed_speed[idx2], pcs[0, :])[0, 1])
                            CC2.append(np.corrcoef(smoothed_speed[idx2], pcs[1, :])[0, 1])
                            for n in range(pcs.shape[1]):
                                if len(PC1) < n+1:
                                    PC1.append(pcs[0, n])
                                    PC2.append(pcs[1, n])
                                    V.append(smoothed_speed[idx2[n]])
                                    Rt.append(np.mean(rate_chunk[:, n]))
                                    PC_n.append(1)
                                else:
                                    PC1[n] += pcs[0, n]
                                    PC2[n] += pcs[1, n]
                                    V[n] += smoothed_speed[idx2[n]]
                                    Rt[n] += np.mean(rate_chunk[:, n])
                                    PC_n[n] += 1
                            i = gap_idx + 1

                            # calculate population statistics
                            Rn.append(np.mean(rate_chunk, axis=1))

                        # calculate averages over time windows
                        pr = np.mean(PR, axis=0)
                        cc1 = np.mean(CC1, axis=0)
                        cc2 = np.mean(CC2, axis=0)
                        pcn = np.asarray(PC_n)
                        pc1, pc2 = np.asarray(PC1) / pcn, np.asarray(PC2) / pcn
                        v = np.asarray(V) / pcn
                        C = np.mean(Cs, axis=0)
                        rn = np.mean(Rn, axis=0)
                        rt = np.asarray(Rt) / pcn

                    except (TypeError, ValueError):
                        continue

                    # store results
                    res, signals = results["results"], results["data"]
                    res["drug"].append(drug)
                    res["dose"].append(dose)
                    res["condition"].append(condition)
                    res["mouse"].append(mouse_id)
                    res["neuron_type"].append("D1" if mouse_id in mice["D1"] else "D2")
                    res["mean(r|t)"].append(np.mean(rt))
                    res["std(r|t)"].append(np.std(rt))
                    res["std(r|i)"].append(np.std(rn))
                    res["dimensionality"].append(pr)
                    res["v"].append(np.round((bins[bin] + bins[bin + 1]) / 2, decimals=1))
                    res["p(v)"].append(speed_hist[bin])
                    res["corr(v,pc1)"].append(cc1)
                    res["corr(v,pc2)"].append(cc2)
                    signals["drug"].append(drug)
                    signals["dose"].append(dose)
                    signals["condition"].append(condition)
                    signals["mouse"].append(mouse_id)
                    signals["neuron_type"].append("D1" if mouse_id in mice["D1"] else "D2")
                    signals["C"].append(C)
                    signals["v"].append(np.round((bins[bin] + bins[bin + 1]) / 2, decimals=1))
                    signals["v(t)"].append(v)
                    signals["pc1"].append(pc1)
                    signals["pc2"].append(pc2)

        print(f"Finished processing data for drug = {drug}.")

    # save data
    df = DataFrame.from_dict(results["results"])
    df.to_csv(f"{save_dir}/spn_dimensionality_data.csv")
    pickle.dump(results["data"], open(f"{save_dir}/spn_dimensionality_data.pkl", "wb"))

else:

    df = read_csv(f"/home/rgast/data/parker_data/spn_dimensionality_data.csv")

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
    df.sort_values(["condition", "dose"], inplace=True)
    for key in ["dimensionality"]:
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
                fig.savefig(f"/home/rgast/data/parker_data/figures/{drug}_{key}_d1_d2_split.svg")
    plt.show()
