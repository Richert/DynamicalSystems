import numpy as np
from scipy.io import loadmat
import os
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import svd
from pandas import DataFrame, read_csv
import pickle

# load processed data or re-process data
path = "/mnt/kennedy_lab_data/Parkerlab/neural_data"
save_dir = "/home/rgast/data/parker_data"
spike_field = "dff_traces_5hz"
speed_field = "speed_traces_5hz"

# choose condition
# drugs = ["clozapine", "olanzapine", "xanomeline", "MP10", "haloperidol", "M4PAM",
#          "SCH23390", "SCH39166", "SEP363856", "SKF38393"]
drug = "haloperidol"
dose = "Vehicle"
mouse = "m404"

# meta parameters
max_neurons = 100
sigma_speed = 1
sigma_rate = 1
v_bins = 5
v_max = 10.0
bins = np.round(np.linspace(0.0, 1.0, num=v_bins+1)*v_max, decimals=1)
epsilon = 1e-15
gap_window = 5
std_norm = True

# mouse identity
mice = {"D1":["m085", "m040", "m298", "m404", "f487", "f694", "f857", "f859", "m794", "m797", "m795", "m973",
              "m974", "m659", "m975", "f976", "f977", "f979"],
        "D2": ["m971", "m972", "m106", "m120", "m377", "m380", "f414", "f480", "m483", "m485", "m241", "m242",
               "m523", "f605", "f808", "f811", "f840"]}
spn_type = 'D1' if mouse in mice['D1'] else 'D2'

# analysis
##########

# load data
data = {}
for file in os.listdir(f"{path}/{drug}/{dose}"):

    if mouse in file:

        condition = "amph" if "amph" in file else "veh"
        data_tmp = loadmat(f"{path}/{drug}/{dose}/{file}/{condition}_drug.mat", simplify_cells=True)
        data[condition] = {"s_raw": data_tmp[f"{condition}_drug"][spike_field],
                           "v_raw": data_tmp[f"{condition}_drug"][speed_field]}

    if ("amph" in data) and ("veh" in data):
        break

else:
    raise FileNotFoundError(f"No data found for {mouse} in directory {path}/{drug}/{dose}.")

for condition in ["veh", "amph"]:

    # get condition data
    data_tmp = data[condition]
    s, v = data_tmp["s_raw"], data_tmp["v_raw"]
    s = s[:max_neurons, :len(v)]

    # calculate smooth variables
    data_tmp["s_smooth"] = np.asarray([gaussian_filter1d(s[i, :], sigma=sigma_rate) for i in range(s.shape[0])])
    data_tmp["v_smooth"] = gaussian_filter1d(v, sigma=sigma_speed)
    s2, v2 = data_tmp["s_smooth"], data_tmp["v_smooth"]

    # calculate neural covariance matrix
    s_centered = np.zeros_like(s2)
    for n in range(s2.shape[0]):
        s_centered[n, :] = s2[n, :] - np.mean(s2[n, :])
        if std_norm:
            s_centered[n, :] /= (np.std(s_centered[n, :]) + epsilon)
    C = np.cov(s_centered, ddof=0)

    # get eigenvalues and eigenvectors of C
    eigvals, eigvecs = np.linalg.eigh(C)
    eig_idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[eig_idx], eigvecs[:, eig_idx]
    pr = np.sum(eigvals)**2 / (np.sum(eigvals**2)*C.shape[0])
    pcs = eigvecs.T @ s_centered

    # get speed histogram
    v_hist = np.histogram(v2, bins=bins)[0]
    v_hist = v_hist / np.sum(v_hist)

    # calculate result variables and save data
    bin_data = {"v": [], "r": [], "C": [], "PC1": [], "v(t)": [], "d": [], "r_dist": []}
    for bin in range(v_bins):

        # find time windows where mouse runs at particular velocity
        idx = (bins[bin] <= v2) & (v2 < bins[bin + 1])
        v_indices = np.argwhere(idx == True).squeeze()
        idx_diff = np.diff(v_indices)

        # go over different time bins
        PR, PC1, PC_n, V, Cs, Rn, Rt = [], [], [], [], [], [], []
        i = 0
        for gap_idx in np.argwhere(idx_diff > gap_window).squeeze():

            # find time window of consistent velocity
            idx2 = v_indices[i:gap_idx+1]
            if len(idx2) < gap_window:
                continue

            # center firing rates
            s_chunk = s2[:, idx2]
            s_c2 = np.zeros_like(s_chunk)
            for n in range(s_chunk.shape[0]):
                s_c2[n, :] = s_chunk[n, :] - np.mean(s_chunk[n, :])
                if std_norm:
                    s_c2[n, :] /= (np.std(s_chunk[n, :]) + epsilon)

            # get covariance matrix and calculate dimensionality
            C2 = np.cov(s_c2)
            eigvals2, eigvecs2 = np.linalg.eigh(C2)
            eig_idx2 = np.argsort(eigvals2)[::-1]
            eigvals2, eigvecs2 = eigvals2[eig_idx2], eigvecs2[:, eig_idx2]
            pr2 = np.sum(eigvals2) ** 2 / (np.sum(eigvals2 ** 2) * C2.shape[0])
            if not np.isfinite(pr2):
                continue
            Cs.append(C2)
            PR.append(pr2)

            # get correlation between PCs and velocity
            pcs2 = eigvecs2.T @ s_c2
            for n in range(pcs2.shape[1]):
                if len(PC1) < n+1:
                    PC1.append(pcs2[0, n])
                    V.append(v2[idx2[n]])
                    Rt.append(np.mean(s_chunk[:, n]))
                    PC_n.append(1)
                else:
                    PC1[n] += pcs2[0, n]
                    V[n] += v2[idx2[n]]
                    Rt[n] += np.mean(s_chunk[:, n])
                    PC_n[n] += 1
            i = gap_idx + 1

            # calculate population statistics
            Rn.append(np.mean(s_chunk, axis=1))

        # calculate averages over time windows
        pr2 = np.mean(PR, axis=0)
        pcn = np.asarray(PC_n)
        pc1 = np.asarray(PC1) / pcn
        v3 = np.asarray(V) / pcn
        C2 = np.mean(Cs, axis=0)
        C2[np.eye(C2.shape[0]) > 0.0] = 0.0
        rn = np.mean(Rn, axis=0)
        rt = np.asarray(Rt) / pcn

        # save bin-based data
        bin_data["v"].append((bins[bin] + bins[bin+1])*0.5)
        bin_data["v(t)"].append(v3)
        bin_data["r"].append(rt)
        bin_data["PC1"].append(pc1)
        bin_data["C"].append(C2)
        bin_data["d"].append(pr2)

    # save data
    C[np.eye(C.shape[0]) > 0.0] = 0.0
    data_tmp["bin_data"] = bin_data
    data_tmp["C"] = C
    data_tmp["d"] = pr
    data_tmp["PC1"] = pcs[0, :] * eigvals[0]
    data_tmp["PC2"] = pcs[1, :] * eigvals[1]
    data_tmp["PC3"] = pcs[2, :] * eigvals[2]
    data_tmp["v_hist"] = v_hist
    data_tmp["eigvals"] = eigvals

# plotting
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams["font.size"] = 12
matplotlib.rcParams["axes.labelsize"] = 12
matplotlib.rcParams["lines.markersize"] = 12.0
matplotlib.rcParams["lines.linewidth"] = 1.5
import seaborn as sb

sb.set_palette("colorblind")

for condition in ["veh", "amph"]:

    data_tmp = data[condition]

    # figure 1
    ##########

    fig = plt.figure(figsize=(18, 8), layout="constrained")
    grid = fig.add_gridspec(nrows=6, ncols=4)
    fig.suptitle(f"Trial-based results, condition: {condition} (drug: {drug}, dose: {dose}, mouse: {mouse})")

    # velocity dynamics
    ax = fig.add_subplot(grid[:2, :3])
    ax.plot(data_tmp["v_raw"], label="raw")
    ax.plot(data_tmp["v_smooth"], label="filtered")
    ax.legend()
    ax.set_ylabel("v (cm/s)")
    ax.set_title("mouse velocity")

    # SPN dynamics
    ax = fig.add_subplot(grid[2:4, :3])
    ax.plot(np.mean(data_tmp["s_raw"], axis=0), label="raw")
    ax.plot(np.mean(data_tmp["s_smooth"], axis=0), label="filtered")
    ax.legend()
    ax.set_ylabel("r (Hz)")
    ax.set_title(f"{spn_type}-SPN dynamics")

    # PC dynamics
    ax = fig.add_subplot(grid[4:, :3])
    ax.plot(data_tmp["PC1"], label="PC1")
    ax.plot(data_tmp["PC2"], label="PC2")
    ax.plot(data_tmp["PC3"], label="PC3")
    ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel("r (Hz)")
    ax.set_title(f"principal component dynamics")

    # covariance matrix
    ax = fig.add_subplot(grid[:4, 3])
    im = ax.imshow(data_tmp["C"], aspect="auto", interpolation="none", cmap="magma")
    ax.set_xlabel("neuron")
    ax.set_ylabel("neuron")
    plt.colorbar(im, ax=ax)
    ax.set_title(f"{spn_type}-SPN Covariance (d = {np.round(data_tmp['d'], decimals=2)})")

    # EV distribution
    ax = fig.add_subplot(grid[4, 3])
    ax.bar(np.arange(0, len(data_tmp["eigvals"])), data_tmp["eigvals"])
    ax.set_xlabel("index")
    ax.set_ylabel("lambda")
    ax.set_title("eigenvalue distribution")

    # velocity distribution
    ax = fig.add_subplot(grid[5, 3])
    ax.bar(np.asarray([(bins[i] + bins[i+1])*0.5 for i in range(v_bins)]), data_tmp["v_hist"])
    ax.set_xlabel("v (cm/s)")
    ax.set_ylabel("p(v)")
    ax.set_title("velocity distribution")

    # figure 2
    ##########

    bin_data = data_tmp["bin_data"]
    fig2 = plt.figure(figsize=(18, 8), layout="constrained")
    grid = fig2.add_gridspec(nrows=3, ncols=3)
    fig2.suptitle(f"Velocity bin-based results, condition: {condition} (drug: {drug}, dose: {dose}, mouse: {mouse})")

    # velocity dynamics per bin
    ax = fig2.add_subplot(grid[0, 0])
    for i, v in enumerate(bin_data["v"]):
        ax.plot(bin_data["v(t)"][i], label=v)
    ax.legend()
    ax.set_ylabel("v")
    ax.set_title("velocity dynamics")

    # rate dynamics per bin
    ax = fig2.add_subplot(grid[1, 0])
    for i, v in enumerate(bin_data["v"]):
        ax.plot(bin_data["r"][i], label=v)
    ax.legend()
    ax.set_ylabel("r")
    ax.set_title("rate dynamics")

    # pc1 dynamics per bin
    ax = fig2.add_subplot(grid[2, 0])
    for i, v in enumerate(bin_data["v"]):
        ax.plot(bin_data["PC1"][i], label=v)
    ax.legend()
    ax.set_ylabel("pc1")
    ax.set_title("PC1 dynamics")
    ax.set_xlabel("time")

    # dimensionality distribution
    ax = fig2.add_subplot(grid[0, 1])
    ax.plot(bin_data["v"], bin_data["d"])
    ax.set_xlabel("v (cm/s)")
    ax.set_ylabel("d")
    ax.set_title("dimensionality")

    # covariance matrices
    for v, C, g in zip(bin_data["v"], bin_data["C"], [grid[1, 1], grid[2, 1], grid[0, 2], grid[1, 2], grid[2, 2]]):
        ax = fig2.add_subplot(g)
        im = ax.imshow(C, aspect="auto", interpolation="none", cmap="magma")
        plt.colorbar(im, ax=ax)
        ax.set_title(f"v = {np.round(v, decimals=1)}")

fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05, hspace=0.01, wspace=0.01)
fig2.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05, hspace=0.01, wspace=0.01)
fig.canvas.draw()
fig2.canvas.draw()
plt.show()