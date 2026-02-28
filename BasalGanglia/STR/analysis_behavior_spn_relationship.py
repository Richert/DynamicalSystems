import numpy as np
from scipy.io import loadmat
import os
from scipy.ndimage import gaussian_filter1d
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
path = "/mnt/kennedy_labdata/Parkerlab/neural_data"
save_dir = "/home/richard/data/parker_data"

# plotting parameters
plot_results = False
d12_combined = False

# choose condition
drugs = ["clozapine",
         "olanzapine", "xanomeline", "MP10", "haloperidol", "M4PAM", "SCH23390", "SCH39166", "SEP363856", "SKF38393"
         ]
spike_field = "dff_traces_5hz"
speed_field = "speed_traces_5hz"

# meta parameters
max_neurons = 100
sigma = 1
v_bins = 5
v_max = 10.0
bins = np.round(np.linspace(0.0, 1.0, num=v_bins+1)*v_max, decimals=1)
epsilon = 1e-15
gap_window = 5
norm_var = False

# mouse identity
mice = {"D1":["m085", "m040", "m298", "m404", "f487", "f694", "f857", "f859", "m794", "m797", "m795", "m973",
              "m974", "m659", "m975", "f976", "f977", "f979"],
        "D2": ["m971", "m972", "m106", "m120", "m377", "m380", "f414", "f480", "m483", "m485", "m241", "m242",
               "m523", "f605", "f808", "f811", "f840"]}

# analysis
##########

res = {"drug": [], "dose": [], "condition": [], "mouse": [], "neuron_type": [],
        "mean(r)": [], "D(r)": [], "D(C)": [],"D_b(C)": [], "std(pc1)": [], "std(pc2)": [],
        "v": [], "p(v)": [], "corr(v,pc1)": [], "corr(v,pc2)": []}
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
            neurons = np.minimum(max_neurons, spikes.shape[0])
            smoothed_spikes = np.asarray([gaussian_filter1d(spikes[i, :len(speed)], sigma=sigma) for i in range(neurons)])
            smoothed_speed = gaussian_filter1d(speed, sigma=sigma)

            # get speed histogram
            speed_hist = np.histogram(smoothed_speed, bins=bins)[0]
            speed_hist = speed_hist / np.sum(speed_hist)

            # calculate result variables and save data
            bin_data = {"v": [], "r": [], "C": [], "PC1": [], "v(t)": [], "d": [], "r_dist": []}
            for bin in range(v_bins):

                # bin-based analyses
                try:

                    # find time windows where mouse runs at particular velocity
                    idx = (bins[bin] <= smoothed_speed) & (smoothed_speed < bins[bin + 1])
                    v_indices = np.argwhere(idx == True).squeeze()
                    idx_diff = np.diff(v_indices)

                    # get eigenvalues and eigenvectors of C
                    pr, eigvals, eigvecs, C, rates_c = get_eigs(smoothed_spikes[:, idx], normalize_variance=norm_var)

                    # go over different time bins
                    PR, CC1, CC2, PC1, PC2, PC_n, Rn = [], [], [], [], [], [], []
                    i = 0
                    for gap_idx in np.argwhere(idx_diff > gap_window).squeeze():

                        # find time window of consistent velocity
                        idx2 = v_indices[i:gap_idx+1]
                        if len(idx2) < gap_window:
                            continue

                        # get covariance statistics
                        pr_b, eigvals_b, eigvecs_b, _, rates_b = get_eigs(smoothed_spikes[:, idx2], normalize_variance=norm_var)
                        PR.append(pr_b)

                        # get correlation between PCs and velocity
                        pcs = eigvecs_b.T @ smoothed_spikes[:, idx2]
                        CC1.append(np.corrcoef(smoothed_speed[idx2], pcs[0, :])[0, 1])
                        CC2.append(np.corrcoef(smoothed_speed[idx2], pcs[1, :])[0, 1])
                        for n in range(pcs.shape[1]):
                            if len(PC1) < n+1:
                                PC1.append(pcs[0, n])
                                PC2.append(pcs[1, n])
                                PC_n.append(1)
                            else:
                                PC1[n] += pcs[0, n]
                                PC2[n] += pcs[1, n]
                                PC_n[n] += 1
                        i = gap_idx + 1

                        # calculate population statistics
                        Rn.append(np.mean(smoothed_spikes[:, idx2], axis=1))

                    # calculate averages over time windows
                    pr_b = np.mean(PR, axis=0)
                    cc1 = np.mean(CC1, axis=0)
                    cc2 = np.mean(CC2, axis=0)
                    pcn = np.asarray(PC_n)
                    pc1, pc2 = np.asarray(PC1) / pcn, np.asarray(PC2) / pcn
                    rn = np.mean(Rn, axis=0)

                except (TypeError, ValueError):
                    continue

                # store results
                if np.isfinite(pr_b):
                    res["drug"].append(drug)
                    res["dose"].append(dose)
                    res["condition"].append(condition)
                    res["mouse"].append(mouse_id)
                    res["neuron_type"].append("D1" if mouse_id in mice["D1"] else "D2")
                    res["mean(r)"].append(np.mean(rn))
                    res["D(r)"].append(np.sum(rn)**2/(np.sum(rn**2)*len(rn)))
                    res["D(C)"].append(pr)
                    res["D_b(C)"].append(pr_b)
                    res["v"].append(np.round((bins[bin] + bins[bin + 1]) / 2, decimals=1))
                    res["p(v)"].append(speed_hist[bin])
                    res["corr(v,pc1)"].append(cc1)
                    res["corr(v,pc2)"].append(cc2)
                    res["std(pc1)"].append(np.std(pc1))
                    res["std(pc2)"].append(np.std(pc2))

    print(f"Finished processing data for drug = {drug}.")

# save data
df = DataFrame.from_dict(res)
cond_str = f"sigma_{int(sigma)}_{'std_norm' if norm_var else ''}"
df.to_csv(f"{save_dir}/spn_dimensionality_{cond_str}.csv")

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
    for key in ["D(C)", "D_b(C)", "D(r)"]:
        for drug in drugs:
            if d12_combined:
                fig, ax = plt.subplots(figsize=(10, 6))
                sb.lineplot(df.loc[df.loc[:, "drug"] == drug, :],
                            x="v", y=key, hue="dose", style="condition", ax=ax,
                            markers=True, dashes=True, err_style='bars', errorbar="se")
                ax.set_title(drug)
                plt.tight_layout()
                fig.canvas.draw()
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
    plt.show()
