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
delta = data["Delta"]
sd = data["SD"]

# analysis parameters
n_cycles = 5
hopf_start = 100
hopf_width = 30
hopf_height = 0.1
sigma_lowpass = 20
threshold = 0.12

# detect bifurcations in time series
####################################

for key in ["lorentz", "gauss"]:

    # filter data
    filtered = gaussian_filter1d(data[key], sigma=sigma_lowpass)

    # find fold bifurcation points
    indices = np.argwhere(filtered > threshold)
    if len(indices) > 1 and indices[-1] < len(filtered) - 1:
        idx_l = indices[0]
        idx_r = indices[-1]
        I_l = I_ext[idx_r]
        I_r = I_ext[idx_l]
        data[f"{key}_fold"] = {"idx": (idx_l, idx_r), "I_ext": (I_l, I_r), "delta": delta, "sd": sd}
    else:
        data[f"{key}_fold"] = {}

    # find hopf bifurcation points
    hbs, _ = find_peaks(filtered, width=hopf_width, prominence=hopf_height)
    hbs = hbs[hbs >= hopf_start]
    if len(hbs) > n_cycles:
        idx_l, idx_r = hbs[0], hbs[-1]
        I_r = I_ext[idx_r]
        I_l = I_ext[idx_l]
        data[f"{key}_hopf"] = {"idx": (idx_l, idx_r), "I_ext": (I_l, I_r), "delta": delta, "sd": sd}
    else:
        data[f"{key}_hopf"] = {}

# save results
##############

pickle.dump(data, open(f"{path}/bifurcations_{neuron_type}_{idx}.pkl", "wb"))

# plotting
##########

# fig, axes = plt.subplots(nrows=3, figsize=(12, 6))
# ax1 = axes[2]
# ax1.plot(I_ext)
# ax1.set_xlabel("time")
# ax1.set_ylabel("I")
# ax1.set_title("input")
# for ax, key, alpha in zip(axes[:2], ["lorentz", "gauss"], [1.0, 0.5]):
#     ax.plot(data[key], color="black")
#     if data[f"{key}_fold"]:
#         bf = f"{key}_fold"
#         x1 = data[bf]["idx"][0]
#         x2 = data[bf]["idx"][-1]
#         ax.axvline(x=x1, color='red', linestyle='--', label=f"I_lp1 = {data[bf]['I_ext'][0]}")
#         ax.axvline(x=x2, color='red', linestyle='--', label=f"I_lp2 = {data[bf]['I_ext'][-1]}")
#         ax1.axvline(x=x1, color='red', linestyle='--', label=f"I_lp1 = {data[bf]['I_ext'][0]}", alpha=alpha)
#         ax1.axvline(x=x2, color='red', linestyle='--', label=f"I_lp2 = {data[bf]['I_ext'][-1]}", alpha=alpha)
#     if data[f"{key}_hopf"]:
#         bf = f"{key}_hopf"
#         x1 = data[bf]["idx"][0]
#         x2 = data[bf]["idx"][-1]
#         ax.axvline(x=x1, color='green', linestyle='--', label=f"I_hb1 = {data[bf]['I_ext'][0]}")
#         ax.axvline(x=x2, color='green', linestyle='--', label=f"I_hb2 = {data[bf]['I_ext'][-1]}")
#         ax1.axvline(x=x1, color='green', linestyle='--', label=f"I_hb1 = {data[bf]['I_ext'][0]}", alpha=alpha)
#         ax1.axvline(x=x2, color='green', linestyle='--', label=f"I_hb2 = {data[bf]['I_ext'][-1]}", alpha=alpha)
#     ax.set_xlabel("time")
#     ax.set_ylabel("s")
#     ax.legend()
#     ax.set_title(key)
# plt.suptitle(rf"$\Delta = {delta}$, $\sigma = {sd}$")
# plt.tight_layout()
# plt.show()
