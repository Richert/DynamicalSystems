import pickle
import numpy as np
import sys
from scipy.signal import butter, sosfilt, hilbert, sosfreqz
import matplotlib.pyplot as plt
from pandas import DataFrame


# load data
tdir = "results/rs_entrainment_0.pkl" #sys.argv[-1]
fn = "results/rs_arnold_tongue_0.pkl" #sys.argv[-2]
data = pickle.load(open(fn, "rb"))

# extract relevant stuff from data
sweep = data["sweep"]
W_in = data["W_in"]
res = data["s"]
ko = data["I_ext"].squeeze()
omega = data["omega"] * 1e3
fs = int(np.round(1e3/(data["sr"]*data["dt"]), decimals=0))

# filtering options
print(f"Sampling frequency: {fs}")
f_margin = 0.5
print(f"Frequency band width (Hz): {2*f_margin}")
f_order = 6
f_cutoff = 100

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
results = DataFrame(np.zeros((len(res), 4)), columns=["coh_inp", "plv_inp", "coh_noinp", "plv_noinp"])
dimensionality, covariances = [], []
for i, ik_net in enumerate(res):

    input_neurons = W_in[i][:, 0] > 0
    ik_inp = np.mean(ik_net.loc[:, input_neurons == True].values, axis=-1)
    ik_noinp = np.mean(ik_net.loc[:, input_neurons == False].values, axis=-1)

    # calculate coherence and PLV
    for ik, cond in zip([ik_inp, ik_noinp], ["inp", "noinp"]):

        # scale data
        ik -= np.min(ik)
        ik_max = np.max(ik)
        if ik_max > 0.0:
            ik /= ik_max

        # filter data around driving frequency
        ik_filtered = butter_bandpass_filter(ik, (omega-f_margin*omega, omega+f_margin*omega), fs=fs, order=f_order)
        ik_max = np.max(ik_filtered)
        if ik_max > 0:
            ik_filtered /= ik_max

        # get analytic signals
        ik_phase, ik_env = analytic_signal(ik_filtered[f_cutoff:-f_cutoff])
        ko_phase, ko_env = analytic_signal(ko[f_cutoff:-f_cutoff])

        # calculate plv and coherence
        plv = phase_locking(ik_phase, ko_phase)
        coh = coherence(ik_phase, ko_phase, ik_env, ko_env)

        # test plotting
        # plt.figure(2)
        # plt.plot(ik_filtered, label="ik_f")
        # plt.plot(ko, label="ko")
        # plt.plot(ik, label="ik")
        # plt.title(f"Coh = {coh}, PLV = {plv}")
        # plt.legend()
        # plt.show()

        # store results
        results.loc[i, f"coh_{cond}"] = coh
        results.loc[i, f"plv_{cond}"] = plv

    # calculate dimensionality of network dynamics
    dim, cov = get_dim(ik_net.values)
    dimensionality.append(dim)
    covariances.append(cov)

# save results
data_new = {"Delta": data["Delta"], "omega": omega, "p": data["p"]}
data_new["entrainment"] = results
data_new["dim"] = dimensionality
data_new["cov"] = covariances
data_new["sweep"] = sweep
pickle.dump(data_new, open(tdir, "wb"))
