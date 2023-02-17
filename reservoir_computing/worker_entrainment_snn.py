import pickle
import numpy as np
import sys
from scipy.signal import butter, sosfilt, hilbert, sosfreqz
import matplotlib.pyplot as plt
from pandas import DataFrame


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


def get_dim(s: np.ndarray):
    s = s - np.mean(s)
    s = s / np.std(s)
    cov = s.T @ s
    cov[np.eye(cov.shape[0]) > 0] = 0.0
    eigs = np.abs(np.linalg.eigvals(cov))
    return np.sum(eigs)**2/np.sum(eigs**2), cov


# load data
tdir = sys.argv[-1]
fn = sys.argv[-2] #"results/rs_arnold_tongue_hom.pkl"
data = pickle.load(open(fn, "rb"))

# extract relevant stuff from data
sweep = data["sweep"]
W_in = data["W_in"]
res = data["s"]
ko = data["I_ext"].squeeze()
omega = data["omega"] * 1e3
print(res[0].index[1], res[0].index[0])
fs = int(np.round(1e3*data["sr"]/data["dt"], decimals=0))

# filtering options
print(f"Sampling frequency: {fs}")
f_margin = 0.5
print(f"Frequency band width (Hz): {2*f_margin}")
f_order = 6
f_cutoff = 10000

# Plot the frequency response for a few different orders.
# plt.figure(1)
# for order, omega in zip([4, 8, 4, 8], [2.0, 2.0, 4.0, 4.0]):
#     sos = butter(order, (omega-f_margin*omega, omega+f_margin*omega), fs=fs, output="sos", btype="bandpass")
#     w, h = sosfreqz(sos, worN=12000)
#     plt.plot((fs * 0.5 / np.pi) * w, abs(h), label=f"Order = {order}, omega = {omega}")
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Gain')
# plt.grid(True)
# plt.legend(loc='best')
# plt.show()

# compute phase locking values and coherences
results = DataFrame(np.zeros((W_in[0].shape[0], 4)), columns=["trial", "input_neuron", "coh", "plv"])
dimensionality, covariances = [], []
for i, ik_net in enumerate(res):

    # calculate coherence and PLV
    for neuron in range(ik_net.shape[1]):

        # extract and scale data
        ik = ik_net.loc[:, neuron].values.squeeze()
        ik -= np.min(ik)
        ik /= np.max(ik)
        ko = np.sin(2.0*np.pi*ko)

        # filter data around driving frequency
        ik_filtered = butter_bandpass_filter(ik, (omega-f_margin*omega, omega+f_margin*omega), fs=fs, order=f_order)
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

        # store results
        results.loc[(i+1)*neuron, "trial"] = i
        results.loc[(i + 1) * neuron, "input_neuron"] = W_in[i][neuron] > 0
        results.loc[(i + 1) * neuron, "coh"] = coh
        results.loc[(i + 1) * neuron, "plv"] = plv

    # calculate dimensionality of network dynamics
    dim, cov = get_dim(ik_net.values)
    dimensionality.append(dim)
    covariances.append(cov)

# save results
data_new = {"Delta": data["delta"], "omega": omega, "p": data["p"]}
data_new["entrainment"] = results
data_new["dim"] = dimensionality
data_new["cov"] = covariances
data_new["sweep"] = sweep
pickle.dump(data_new, open(tdir, "wb"))
