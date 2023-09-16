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
data = pickle.load(open(f"{path}/bifurcations_{neuron_type}_{idx}.pkl", "rb"))
I_ext = data["I_ext"]

# analysis parameters
cutoff = 10000
n_cycles = 3
hopf_diff = 3000
hopf_len = 5000
hopf_start = 100000
sigma_lowpass = 50

# detect bifurcations in time series
####################################

for key in ["lorentz", "gauss"]:

    # filter data
    filtered = gaussian_filter1d(data[key].squeeze(), sigma=sigma_lowpass)

    # find fold bifurcation points
    lps, props = find_peaks(-1.0 * filtered, width=1000, prominence=0.02)
    if len(lps) > 1:
        I_r = I_ext[lps[0] - int(0.5 * props['widths'][0])]
        I_l = I_ext[lps[-1]]
        data[f"{key}_fold"] = (I_l, I_r)
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

# save results
##############

pickle.dump(data, open(f"{path}/bifurcations_{neuron_type}_{idx}.pkl", "wb"))
