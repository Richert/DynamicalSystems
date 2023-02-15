import pickle
import numpy as np
import sys
from scipy.signal import butter, sosfilt, hilbert, sosfreqz
import matplotlib.pyplot as plt


def butter_bandpass_filter(data: np.ndarray, freqs: tuple, fs: int, order: int) -> np.ndarray:
    sos = butter(order, freqs, btype="bandpass", output="sos", fs=fs)
    return sosfilt(sos, data)


def analytic_signal(sig: np.ndarray) -> tuple:
    sig_analytic = hilbert(sig)
    sig_phase = np.unwrap(np.angle(sig_analytic))
    sig_envelope = np.abs(sig_analytic)
    return sig_phase, sig_envelope


def phase_locking(x: np.ndarray, y: np.ndarray) -> float:
    return np.abs(np.mean(np.exp(1.0j*(x-y))))


def coherence(x_phase: np.ndarray, y_phase: np.ndarray, x_env: np.ndarray, y_env: np.ndarray) -> float:
    coh = np.abs(np.sum(x_env * y_env * np.exp(1.0j*(x_phase - y_phase))))
    return coh / np.sqrt(np.sum(x_env**2) * np.sum(y_env**2))


# load data
fn = sys.argv[-1] #"results/rs_arnold_tongue_hom.pkl"
data = pickle.load(open(fn, "rb"))

# extract relevant stuff from data
alphas = data["alphas"]*1e3
omegas = data["omegas"]*1e3
res = data["res"]
res_map = data["map"]
fs = int(np.round(1000.0/(res.index[1] - res.index[0]), decimals=0))

# filtering options
print(f"Sampling frequency: {fs}")
f_margin = 0.5
print(f"Frequency band width (Hz): {2*f_margin}")
f_order = 8
f_cutoff = 100000

# Plot the frequency response for a few different orders.
plt.figure(1)
for order, omega in zip([4, 8, 4, 8], [2.0, 2.0, 4.0, 4.0]):
    sos = butter(order, (omega-f_margin*omega, omega+f_margin*omega), fs=fs, output="sos", btype="bandpass")
    w, h = sosfreqz(sos, worN=12000)
    plt.plot((fs * 0.5 / np.pi) * w, abs(h), label=f"Order = {order}, omega = {omega}")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid(True)
plt.legend(loc='best')
plt.show()

# compute phase locking values and coherences
coherences = np.zeros((len(alphas), len(omegas)))
plvs = np.zeros_like(coherences)
for key in res_map.index:

    # extract and scale data
    omega = res_map.at[key, 'omega'] * 1e3
    alpha = res_map.at[key, 'alpha'] * 1e3
    ik = res["ik"][key].values.squeeze()
    ik -= np.min(ik)
    ik /= np.max(ik)
    ko = res["ko"][key].values.squeeze()
    ko = np.sin(2.0*np.pi*ko)

    # filter data around driving frequency
    ik_filtered = butter_bandpass_filter(ik, omega, fs=fs, order=f_order)
    ik_filtered /= np.max(ik_filtered)

    # get analytic signals
    ik_phase, ik_env = analytic_signal(ik_filtered[f_cutoff:-f_cutoff])
    ko_phase, ko_env = analytic_signal(ko[f_cutoff:-f_cutoff])

    # calculate plv and coherence
    plv = phase_locking(ik_phase, ko_phase)
    coh = coherence(ik_phase, ko_phase, ik_env, ko_env)

    # test plotting
    plt.figure(2)
    plt.plot(ik_filtered, label="ik_f")
    plt.plot(ko, label="ko")
    plt.plot(ik, label="ik")
    plt.title(f"Coh = {coh}, PLV = {plv}")
    plt.legend()
    plt.show()

    # find matrix position that corresponds to these parameters
    idx_r = np.argmin(np.abs(alphas - alpha))
    idx_c = np.argmin(np.abs(omegas - omega))

    # store coherence value at driving frequency
    plvs[idx_r, idx_c] = plv
    coherences[idx_r, idx_c] = coh

# save results
data["coherence"] = coherences
data["plv"] = plvs
pickle.dump(data, open(fn, "wb"))
