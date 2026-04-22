import numpy as np
from scipy.io import loadmat
import os
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, correlate

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
save_dir = "/home/rgast/data/parker_data"
spike_field = "dff_traces_5hz"
speed_field = "speed_traces_5hz"

# choose condition
# drugs = ["clozapine", "olanzapine", "xanomeline", "MP10", "haloperidol", "M4PAM",
#          "SCH23390", "SCH39166", "SEP363856", "SKF38393"]
drug = "haloperidol"
dose = "Vehicle"
mouse = "m972"
behaviors = {
    "v0": (0.0, 1.0, -0.5, 0.5), "v1": (1.0, 3.0, -0.5, 0.5),
    "v2": (3.0, 6.0, -0.5, 0.5), "v3": (6.0, np.inf, -0.5, 0.5),
    "p0": (1.0, 3.0, 1, 50), "p1": (3.0, 6.0, 1, 50),
    "p2": (6.0, 10.0, 1, 50), "p3": (10.0, np.inf, 1, 50)
}

# meta parameters
max_neurons = 50
sigma_speed = 1
sigma_rate = 1
epsilon = 1e-15
std_norm = False
window_based = False
gap_window = 5
corr_window = 200
spn_window = 3

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
    s = s[:max_neurons, :len(v) - 1]
    data_tmp["a_raw"] = np.diff(v)
    data_tmp["s_raw"] = s
    data_tmp["v_raw"] = v[:-1]

    # calculate smooth variables
    s2 = np.asarray([gaussian_filter1d(s[i, :], sigma=sigma_rate) for i in range(s.shape[0])])
    v2 = gaussian_filter1d(v, sigma=sigma_speed)
    a2 = np.diff(v2)
    v2 = v2[:-1]
    data_tmp["s_smooth"], data_tmp["v_smooth"], data_tmp["a_smooth"] = s2, v2, a2

    # show recurrence plot for velocity
    v_tmp = v2 - np.mean(v2)
    v_tmp /= np.std(v_tmp)
    idx = 0
    v_c = []
    while idx + corr_window < len(v2):
        v_tmp2 = v_tmp[idx:idx+corr_window]
        v_c.append(correlate(v_tmp2, v_tmp2, mode="full"))
        idx += corr_window
    data_tmp["v_c"] = np.mean(v_c, axis=0)

    # calculate neural covariance matrix
    pr, eigvals, eigvecs, C, s_centered = get_eigs(s2, normalize_variance=std_norm)
    pcs = eigvecs.T @ s_centered

    # calculate result variables and save data
    behavior_data = {"b": [], "v": [], "r": [], "C": [], "D(C)": [], "D(r)": [], "r_dist": [], "b_idx": [], "p(b)": []}
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

        if window_based:

            idx_diff = np.diff(1.0 * idx)
            starts, stops = np.argwhere(idx_diff > 0.0).squeeze(), np.argwhere(idx_diff < 0.0).squeeze()
            behavior_data_tmp = {"r": [], "C": [], "D(C)": [], "D(r)": [], "r_dist": []}
            for start, stop in zip(starts, stops):

                if stop-start < spn_window:
                    continue

                # get covariance matrix and calculate dimensionality
                s2_idx = s2[:, start:stop]
                pr2, eigvals2, eigvecs2, C2, s_c2 = get_eigs(s2_idx, normalize_variance=std_norm)
                pcs2 = eigvecs2.T @ s2_idx
                if not np.isfinite(pr2):
                    continue

                # save window data
                r2 = np.mean(s2_idx, axis=1)
                behavior_data_tmp["r"].append(np.mean(r2))
                behavior_data_tmp["C"].append(C2)
                behavior_data_tmp["D(C)"].append(pr2)
                behavior_data_tmp["D(r)"].append(np.sum(r2)**2 / (np.sum(r2**2)*len(r2)))
                behavior_data_tmp["r_dist"].append(r2)

            # save behavior specific data
            behavior_data["b"].append(b)
            behavior_data["v"].append(np.mean(v2[idx]))
            behavior_data["r"].append(np.mean(behavior_data_tmp["r"]))
            behavior_data["C"].append(np.mean(behavior_data_tmp["C"], axis=0))
            behavior_data["D(C)"].append(np.mean(behavior_data_tmp["D(C)"]))
            behavior_data["D(r)"].append(np.mean(behavior_data_tmp["D(r)"]))
            behavior_data["r_dist"].append(np.mean(behavior_data_tmp["r_dist"], axis=0))
            behavior_data["b_idx"].append(idx)
            behavior_data["p(b)"].append(np.mean(idx))

        else:

            # get covariance matrix and calculate dimensionality
            s2_idx = s2[:, idx]
            pr2, eigvals2, eigvecs2, C2, s_c2 = get_eigs(s2_idx, normalize_variance=std_norm)
            pcs2 = eigvecs2.T @ s_c2
            if not np.isfinite(pr2):
                continue

            # save behavior-specific data
            r2 = np.mean(s2_idx, axis=1)
            behavior_data["b"].append(b)
            behavior_data["v"].append(np.mean(v2[idx]))
            behavior_data["r"].append(np.mean(r2))
            behavior_data["C"].append(C2)
            behavior_data["D(C)"].append(pr2)
            behavior_data["D(r)"].append(np.sum(r2)**2/(np.sum(r2**2)*len(r2)))
            behavior_data["r_dist"].append(r2)
            behavior_data["b_idx"].append(idx)
            behavior_data["p(b)"].append(np.mean(idx))

    # save data
    data_tmp["behavior_data"] = behavior_data
    data_tmp["C"] = C
    data_tmp["d"] = pr
    data_tmp["PC1"] = pcs[0, :] * eigvals[0]
    data_tmp["PC2"] = pcs[1, :] * eigvals[1]
    data_tmp["PC3"] = pcs[2, :] * eigvals[2]
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

    # figure 1: trial summary
    #########################

    fig = plt.figure(figsize=(18, 8), layout="constrained")
    grid = fig.add_gridspec(nrows=6, ncols=4)
    fig.suptitle(f"Trial-based results, condition: {condition} (drug: {drug}, dose: {dose}, mouse: {mouse})")

    # velocity dynamics
    ax0 = fig.add_subplot(grid[:2, :3])
    v = data_tmp["v_smooth"]
    ax0.plot(data_tmp["v_raw"], label="raw")
    ax0.plot(v, label="filtered")
    ax0.legend()
    ax0.set_ylabel("v (cm/s)")
    ax0.set_title("mouse velocity")

    # acceleration dynamics
    ax = fig.add_subplot(grid[2:4, :3])
    ax.plot(data_tmp["a_raw"], label="raw")
    ax.plot(data_tmp["a_smooth"], label="filtered")
    ax.legend()
    ax.set_ylabel("a (cm/s2)")
    ax.set_title("mouse acceleration")
    ax.sharex(ax0)

    # SPN dynamics
    ax = fig.add_subplot(grid[4:, :3])
    ax.plot(np.mean(data_tmp["s_raw"], axis=0), label="raw")
    ax.plot(np.mean(data_tmp["s_smooth"], axis=0), label="filtered")
    ax.legend()
    ax.set_ylabel("r (Hz)")
    ax.set_title(f"{spn_type}-SPN dynamics")
    ax.sharex(ax0)

    # covariance matrix
    ax = fig.add_subplot(grid[:4, 3])
    im = ax.imshow(data_tmp["C"], aspect="auto", interpolation="none", cmap="magma")
    ax.set_xlabel("neuron")
    ax.set_ylabel("neuron")
    plt.colorbar(im, ax=ax)
    ax.set_title(f"{spn_type}-SPN Covariance (d = {np.round(data_tmp['d'], decimals=2)})")

    # EV distribution
    ax = fig.add_subplot(grid[4:, 3])
    ax.bar(np.arange(0, len(data_tmp["eigvals"])), data_tmp["eigvals"])
    ax.set_xlabel("index")
    ax.set_ylabel("lambda")
    ax.set_title("eigenvalue distribution")

    # figure 2: behavior-specific analysis
    ######################################

    b_data = data_tmp["behavior_data"]
    fig2 = plt.figure(figsize=(18, 8), layout="constrained")
    grid = fig2.add_gridspec(nrows=3, ncols=3)
    fig2.suptitle(f"Velocity bin-based results, condition: {condition} (drug: {drug}, dose: {dose}, mouse: {mouse})")

    # velocity dynamics and associated behaviors
    ax = fig2.add_subplot(grid[0, :])
    v = data_tmp["v_smooth"]
    ax.plot(v)
    x = np.arange(len(v))
    for i, b in enumerate(b_data["b"]):
        p = b_data["b_idx"][i]
        w = np.zeros_like(v)
        w[p] = 1.0
        ax.fill_between(x=x, y1=0.0, y2=np.max(v), where=w, label=b, alpha=0.6)
    # ax.legend()
    ax.set_ylabel("v (cm/s)")
    ax.set_xlabel("time steps")
    ax.set_title("mouse behavior")

    # velocity autocorrelation
    ax = fig2.add_subplot(grid[1, 0])
    ax.plot(data_tmp["v_c"])
    ax.set_ylabel("corr")
    ax.set_xlabel("time lag")
    ax.set_title("velocity autocorrelation")

    # behavior distribution
    ax = fig2.add_subplot(grid[1, 1])
    fractions = []
    for i, b in enumerate(b_data["b"]):
        fractions.append(b_data["p(b)"][i])
    ax.bar(b_data["b"], fractions, align="center")
    ax.set_ylabel("p")
    ax.set_xlabel("behavior")
    ax.set_title("behavior distribution")

    # rate distribution per behavior
    ax = fig2.add_subplot(grid[1, 2])
    for i, b in enumerate(b_data["b"]):
        ax.hist(b_data["r_dist"][i], label=b, alpha=0.5)
    # ax.legend()
    ax.set_ylabel("pc1")
    ax.set_title("PC1 dynamics")
    ax.set_xlabel("time")

    # dimensionality per behavior
    ax = fig2.add_subplot(grid[2, 0])
    ax.plot(b_data["b"], b_data["D(C)"])
    ax.set_xlabel("behavior")
    ax.set_ylabel("D(C)")
    ax.set_title("SPN dimensionality")

    # firing rate per behavior
    ax = fig2.add_subplot(grid[2, 1])
    ax.plot(b_data["b"], b_data["r"])
    ax.set_xlabel("behavior")
    ax.set_ylabel("r")
    ax.set_title("SPN firing rates")

    # firing rate heterogeneity per behavior
    ax = fig2.add_subplot(grid[2, 2])
    ax.plot(b_data["b"], b_data["D(r)"])
    ax.set_xlabel("behavior")
    ax.set_ylabel("D(r)")
    ax.set_title("SPN rate heterogeneity")

    # figure 3
    ##########

    fig3 = plt.figure(figsize=(16, 6), layout="constrained")
    nc = 4
    grid = fig3.add_gridspec(ncols=nc, nrows=2)
    fig3.suptitle(f"SPN-Behavior Relationship, condition: {condition}")
    for i, b in enumerate(b_data["b"]):

        ax = fig3.add_subplot(grid[i // nc, i % nc])
        try:
            rates = data_tmp["s_smooth"]
            C, pr = b_data["C"][i], b_data["D(C)"][i]
            im = ax.imshow(C, aspect="auto", interpolation="none", cmap="magma")
            plt.colorbar(im, ax=ax)
            ax.set_title(f"Behavior = {b}, D(C) = {np.round(pr, decimals=3)}")
        except TypeError:
            pass

    fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05, hspace=0.05, wspace=0.05)
    fig2.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05, hspace=0.05, wspace=0.05)
    fig3.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05, hspace=0.05, wspace=0.05)
    fig.canvas.draw()
    fig2.canvas.draw()
    fig3.canvas.draw()

plt.show()