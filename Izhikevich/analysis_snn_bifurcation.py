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
idx = int(sys.argv[-3])
neuron_type = str(sys.argv[-2])
path = str(sys.argv[-1])

# load data
data = pickle.load(open(f"{path}/bifurcations_{neuron_type}_{idx}.p", "rb"))
I_ext = data["I_ext"]

# analysis parameters
n_cycles = 3
hopf_diff = 3000
hopf_len = 5000
hopf_start = 100
sigma_lowpass = 50

# detect bifurcations in time series
####################################

for key in ["lorentz", "gauss"]:

    # filter data
    filtered = gaussian_filter1d(data[key], sigma=sigma_lowpass)

    # find fold bifurcation points
    lps, props = find_peaks(-1.0 * filtered, width=1000, prominence=0.02)
    if len(lps) > 1:
        I_r = I_ext[lps[0] - int(0.5 * props['widths'][0])]
        I_l = I_ext[lps[-1]]
        data[f"{key}_fold"] = (I_l, I_r)
        data["fold_width"] = props["width"][0]
    else:
        data[f"{key}_fold"] = ()

    # find hopf bifurcation points
    hbs, _ = find_peaks(filtered, width=100, prominence=0.006)
    hbs = hbs[hbs >= hopf_start]
    if len(hbs) > n_cycles:
        idx = 0
        while idx < len(hbs) - 1:
            idx_l, idx_r, idx = get_hopf_area(hbs, idx, cutoff=hopf_diff)
            if idx_r - idx_l > hopf_len:
                I_r = I_ext[idx_l]
                I_l = I_ext[idx_r]
                data[f"{key}_hopf"] = (I_l, I_r)
            else:
                data[f"{key}_hopf"] = ()
    else:
        data[f"{key}_hopf"] = ()

# save results
##############

pickle.dump(data, open(f"{path}/bifurcations_{neuron_type}_{idx}.pkl", "wb"))

# plotting
##########

fig, axes = plt.subplots(nrows=2, figsize=(12, 6))

for ax, key in zip(axes, ["lorentz", "gauss"]):
    ax.plot(data["lorentz"], color="black")
    if data[f"{key}_fold"]:
        x1 = data[f"{key}_fold"][0] - int(0.5*data['fold_width'])
        x2 = data[f"{key}_fold"][-1]
        ax.axvline(x=x1, color='red', linestyle='--', label=f"I_lp1 = {data['I_ext'][x1]}")
        ax.axvline(x=x2, color='green', linestyle='--', label=f"I_lp2 = {data['I_ext'][x2]}")
    if data[f"{key}_hopf"]:
        x1 = data[f"{key}_hopf"][0]
        x2 = data[f"{key}_hopf"][-1]
        ax.axvline(x=x1, color='red', linestyle='--', label=f"I_hb1 = {data['I_ext'][x1]}")
        ax.axvline(x=x2, color='green', linestyle='--', label=f"I_hb2 = {data['I_ext'][x2]}")
    ax.set_xlabel("time")
    ax.set_ylabel("s")
    ax.set_title(key)
plt.tight_layout()
plt.show()
