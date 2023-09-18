from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import pickle
import numpy as np
import sys


def get_hopf_area(hbs: np.ndarray, idx: int, cutoff: int):
    diff = np.diff(hbs)
    idx_l = hbs[idx]
    idx_r = idx_l
    while idx < len(diff):
        idx_r = hbs[idx]
        idx += 1
        if diff[idx-1] > cutoff:
            break
    return idx_l, idx_r, idx


# preparations
##############

# define condition
idx = 5 #int(sys.argv[-3])
neuron_type = "fs" #str(sys.argv[-2])
path = "results" #str(sys.argv[-1])

# load data
data = pickle.load(open(f"{path}/bifurcations_{neuron_type}_{idx}.p", "rb"))
I_ext = data["I_ext"]

# analysis parameters
n_cycles = 3
hopf_diff = 3000
hopf_len = 5000
hopf_start = 100
sigma_lowpass = 100
threshold = 0.1

# detect bifurcations in time series
####################################

for key in ["lorentz", "gauss"]:

    # filter data
    filtered = gaussian_filter1d(data[key], sigma=sigma_lowpass)

    # find fold bifurcation points
    indices = np.argwhere(filtered > threshold)
    if len(indices) > 1:
        idx_l = indices[0]
        idx_r = indices[-1]
        I_r = I_ext[idx_l]
        I_l = I_ext[idx_r]
        data[f"{key}_fold"] = {"idx": (idx_l, idx_r), "I_ext": (I_l, I_r)}
    else:
        data[f"{key}_fold"] = {}

    # find hopf bifurcation points
    hbs, _ = find_peaks(-1.0 * filtered, width=1000, prominence=0.05)
    hbs = hbs[hbs >= hopf_start]
    if len(hbs) > n_cycles:
        idx = 0
        while idx < len(hbs) - 1:
            idx_l, idx_r, idx = get_hopf_area(hbs, idx, cutoff=hopf_diff)
            if idx_r - idx_l > hopf_len:
                I_r = I_ext[idx_l]
                I_l = I_ext[idx_r]
                data[f"{key}_hopf"] = {"idx": (idx_l, idx_r), "I_ext": (I_l, I_r)}
            else:
                data[f"{key}_hopf"] = {}
    else:
        data[f"{key}_hopf"] = {}

# save results
##############

pickle.dump(data, open(f"{path}/bifurcations_{neuron_type}_{idx}.pkl", "wb"))

# plotting
##########

fig, axes = plt.subplots(nrows=2, figsize=(12, 6))

for ax, key in zip(axes, ["lorentz", "gauss"]):
    ax.plot(data[key], color="black")
    if data[f"{key}_fold"]:
        bf = f"{key}_fold"
        x1 = data[bf]["idx"][0]
        x2 = data[bf]["idx"][-1]
        ax.axvline(x=x1, color='red', linestyle='--', label=f"I_lp1 = {data[bf]['I_ext'][0]}")
        ax.axvline(x=x2, color='red', linestyle='--', label=f"I_lp2 = {data[bf]['I_ext'][-1]}")
    if data[f"{key}_hopf"]:
        bf = f"{key}_hopf"
        x1 = data[bf]["idx"][0]
        x2 = data[bf]["idx"][-1]
        ax.axvline(x=x1, color='green', linestyle='--', label=f"I_hb1 = {data[bf]['I_ext'][0]}")
        ax.axvline(x=x2, color='green', linestyle='--', label=f"I_hb2 = {data[bf]['I_ext'][-1]}")
    ax.set_xlabel("time")
    ax.set_ylabel("s")
    ax.legend()
    ax.set_title(key)
plt.tight_layout()
plt.show()
