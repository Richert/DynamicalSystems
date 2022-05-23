from os import walk
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import pickle
import numpy as np


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


dir = 'results/eic2'
cutoff = 10000
n_cycles = 3
hopf_diff = 3000
hopf_len = 5000
hopf_start = 100000
_, _, fnames = next(walk(dir), (None, None, []))
lp1s, lp2s, hb1s, hb2s = [], [], [], []
for f in fnames:

    # extract data
    data = pickle.load(open(f"{dir}/{f}", "rb"))
    Is = data['I'][cutoff:]
    res = data['results']
    delta = data['delta_i']

    # filter data
    filtered = gaussian_filter1d(data['results']['ui'].squeeze(), sigma=50)

    # find fold bifurcation points
    lps, props = find_peaks(-1.0*filtered, width=1000, prominence=0.02)
    if len(lps) > 1:
        I_r = Is[lps[0] - int(0.5*props['widths'][0])]
        I_l = Is[lps[-1]]
        lp1s.append([I_l, delta])
        lp2s.append([I_r, delta])

    # find hopf bifurcation points
    hbs, _ = find_peaks(filtered, width=100, prominence=0.006)
    hbs = hbs[hbs >= hopf_start]
    if len(hbs) > n_cycles:
        idx = 0
        while idx < len(hbs)-1:
            idx_l, idx_r, idx = get_hopf_area(hbs, idx, cutoff=hopf_diff)
            if idx_r - idx_l > hopf_len:
                I_r = Is[idx_l]
                I_l = Is[idx_r]
                hb1s.append([I_l, delta])
                hb2s.append([I_r, delta])

# save results
lp1s = np.asarray(lp1s)
lp2s = np.asarray(lp2s)
hb1s = np.asarray(hb1s)
hb2s = np.asarray(hb2s)
pickle.dump({'lp1': lp1s, 'lp2': lp2s, 'hb1': hb1s, 'hb2': hb2s}, open("results/eic_results2.p", "wb"))

# plot results
fig, ax = plt.subplots()
ax.scatter(lp1s[:, 0], lp1s[:, 1], c='grey', s=3, marker='.')
ax.scatter(lp2s[:, 0], lp2s[:, 1], c='grey', s=3, marker='.')
ax.scatter(hb1s[:, 0], hb1s[:, 1], c='green', s=3, marker='.')
ax.scatter(hb2s[:, 0], hb2s[:, 1], c='green', s=3, marker='.')
plt.show()
