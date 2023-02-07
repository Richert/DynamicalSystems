import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, sosfilt, coherence
import pickle
import numpy as np


# load data
###########

data = pickle.load(open("results/rs_arnold_tongue.pkl", "rb"))
alphas = data["alphas"]
omegas = data["omegas"]
res = data["res"]
res_map = data["map"]
dts = res.index[1] - res.index[0]

# coherence calculation
#######################

# calculate and store coherences
coherences = np.zeros((len(alphas), len(omegas)))
nps = 16000
window = 'hamming'
width = 0.3
cutoff = 100000
for key in res_map.index:

    # extract parameter set
    omega = res_map.at[key, 'omega']
    alpha = res_map.at[key, 'alpha']

    # calculate coherence
    freq, coh = coherence(res['ik'][key].squeeze().values, np.sin(2 * np.pi * res['ko'][key].squeeze().values), fs=1/dts, nperseg=nps, window=window)

    # find coherence matrix position that corresponds to these parameters
    idx_r = np.argmin(np.abs(alphas - alpha))
    idx_c = np.argmin(np.abs(omegas - omega))

    # store coherence value at driving frequency
    tf = freq[np.argmin(np.abs(freq - omega))]
    coherences[idx_r, idx_c] = np.max(coh[(freq >= tf-width*tf) * (freq <= tf+width*tf)])

# plot the coherence at the driving frequency for each pair of omega and J
fix, ax = plt.subplots(figsize=(12, 8))
cax = ax.imshow(coherences[::-1, :], aspect='equal', interpolation="none")
ax.set_xlabel(r'$\omega$ (Hz)')
ax.set_ylabel(r'$\alpha$ (Hz)')
ax.set_xticks(np.arange(0, len(omegas), 3))
ax.set_yticks(np.arange(0, len(alphas), 3))
ax.set_xticklabels(np.round(omegas[::3]*1e3, decimals=0))
ax.set_yticklabels(np.round(alphas[::-3]*1e3, decimals=0))
plt.title("Coherence between IK population and KO")
plt.colorbar(cax)
plt.show()