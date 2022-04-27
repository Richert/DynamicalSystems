from os import walk
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import pickle
import numpy as np

dir = 'results/sfa'
cutoff = 10000
n_cycles = 2
_, _, fnames = next(walk(dir), (None, None, []))
lp1s, lp2s, hb1s, hb2s = [], [], [], []
for f in fnames:

    # extract data
    data = pickle.load(open(f"{dir}/{f}", "rb"))
    Is = data['I'][cutoff:]
    res = data['results']
    d = data['d']

    # find fold bifurcation points
    lps, _ = find_peaks(-1.0 * res['u'].squeeze(), width=4000, prominence=2.0)
    if len(lps) > 1:
        I_r = Is[lps[0]]
        I_l = Is[lps[-1]]
        lp1s.append([I_l, d])
        lp2s.append([I_r, d])

    # find hopf bifurcation points
    hbs, _ = find_peaks(-1.0 * res['u'].squeeze(), width=500, prominence=1.0)
    if len(hbs) > len(lps)+n_cycles:
        hbs = hbs[hbs > lps[0]]
        hbs = hbs[hbs < 100000]
        I_r = Is[hbs[0]]
        I_l = Is[hbs[-n_cycles]]
        hb1s.append([I_l, d])
        hb2s.append([I_r, d])

# save results
lp1s = np.asarray(lp1s)
lp2s = np.asarray(lp2s)
hb1s = np.asarray(hb1s)
hb2s = np.asarray(hb2s)
pickle.dump({'lp1': lp1s, 'lp2': lp2s, 'hb1': hb1s, 'hb2': hb2s}, open("results/sfa_results.p", "wb"))

# plot results
fig, ax = plt.subplots()
ax.scatter(lp1s[:, 0], lp1s[:, 1], c='grey', s=3, marker='.')
ax.scatter(lp2s[:, 0], lp2s[:, 1], c='grey', s=3, marker='.')
ax.scatter(hb1s[:, 0], hb1s[:, 1], c='green', s=3, marker='.')
ax.scatter(hb2s[:, 0], hb2s[:, 1], c='green', s=3, marker='.')
plt.show()
