from os import walk
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import pickle
import numpy as np

# example code for a single file
################################

# # load data
# data = pickle.load(open(f"results/sfa/sfa_rnn_15.p", "rb"))
# print(f"d = {data['d']}")
#
# # peak detection
# peaks, properties = find_peaks(-1.0*data['results']['u'].squeeze(), width=500, prominence=2.0)
#
# # plotting
# fig, ax = plt.subplots(nrows=2)
# ax[0].plot(data['results']['u'])
# for i in range(len(peaks)):
#     p = properties['right_ips'][i]
#     ax[0].axvline(x=p, color='orange', linestyle='--')
# ax[1].plot(data['I'])
# plt.show()

# actual analysis
#################

dir = 'results/sfa'
_, _, fnames = next(walk(dir), (None, None, []))
lp1s, lp2s, hb1s, hb2s = [], [], [], []
for f in fnames:

    # extract data
    data = pickle.load(open(f"{dir}/{f}", "rb"))
    Is = data['I']
    res = data['results']
    d = data['d']

    # run peak detection
    peaks, properties = find_peaks(-1.0 * res['u'].squeeze(), width=500, prominence=1.0)

    # find fold bifurcation points
    if len(peaks) > 1:
        I_r = Is[int(properties['right_ips'][0])]
        if len(peaks) < 3:
            I_l = Is[int(properties['right_ips'][-1])]
            lp1s.append([I_l, d])
            lp2s.append([I_r, d])
        else:
            I_l = Is[int(properties['right_ips'][-2])]
            hb1s.append([I_l, d])
            hb2s.append([I_r, d])

# save results
lp1s = np.asarray(lp1s)
lp2s = np.asarray(lp2s)
hb1s = np.asarray(hb1s)
hb2s = np.asarray(hb2s)
pickle.dump({'lp1': lp1s, 'lp2': lp2s, 'hb1': hb1s, 'hb2': hb2s}, open("results/sfa_results.p", "wb"))
