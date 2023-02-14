import pickle
import numpy as np
import sys
from scipy.signal import butter, lfilter, hilbert


def butter_bandpass(lowcut: float, highcut: float, fs: int, order: int = 5) -> tuple:
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', output='ba')
    return b, a


def butter_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 5) -> np.ndarray:
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def ztransform(x: np.ndarray) -> np.ndarray:
    x -= np.mean(x)
    x /= np.std(x)
    return x


def analytic_signal(sig: np.ndarray) -> tuple:
    sig_analytic = hilbert(sig)
    sig_phase = np.unwrap(np.angle(sig_analytic))
    sig_envelope = np.abs(sig_analytic)
    return sig_phase, sig_envelope


def phase_locking(x: np.ndarray, y: np.ndarray) -> float:
    return np.abs(np.sum(np.exp(0.0+1.0j*(x-y)))) / len(x)


def coherence(x_phase: np.ndarray, y_phase: np.ndarray, x_env: np.ndarray, y_env: np.ndarray) -> float:
    coh = np.sum(x_env * y_env * np.exp(0.0+1.0j*(x_phase - y_phase)))
    return coh / np.sqrt(np.sum(x_env**2) + np.sum(y_env**2))


# load data
fn = sys.argv[-1]
data = pickle.load(open(fn, "rb"))

# extract relevant stuff from data
alphas = data["alphas"]
omegas = data["omegas"]
res = data["res"]
res_map = data["map"]
fs = int(1/(res.index[1] - res.index[0]))

# filtering options
print(f"Sampling frequency: {fs}")
f_margin = 0.75*np.min(np.diff(omegas))
print(f"Frequency band width: {2*f_margin}")
f_order = 9

# compute phase locking values and coherences
coherences = np.zeros((len(alphas), len(omegas)))
plvs = np.zeros_like(coherences)
for key in res_map.index:

    # extract data
    omega = res_map.at[key, 'omega']
    alpha = res_map.at[key, 'alpha']
    ik = res["ik"][key].values.squeeze()
    ko = res["ko"][key].values.squeeze()

    # filter data around driving frequency
    ik_filtered = butter_bandpass_filter(ik, omega-f_margin, omega+f_margin, fs=fs, order=f_order)

    # get analytic signals
    ik_phase, ik_env = analytic_signal(ik_filtered)
    ko_phase, ko_env = analytic_signal(ko)

    # calculate plv and coherence
    plv = phase_locking(ik_phase, ko_phase)
    coh = coherence(ik_phase, ko_phase, ik_env, ko_env)

    # test plotting
    import matplotlib.pyplot as plt
    plt.plot(ik_filtered, label="ik_f")
    plt.plot(ko, label="ko")
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
