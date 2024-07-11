from rectipy import Network, circular_connectivity
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import poisson, rv_discrete
from scipy.signal import find_peaks
import sys
from custom_functions import *

# define parameters
###################

# get sweep condition
rep = 0 #int(sys.argv[-1])
g = 10.0 #float(sys.argv[-2])
Delta = 3.0 #float(sys.argv[-3])

# model parameters
N = 1000
p = 0.2
sigma = 50.0
C = 50.0
k = 1.0
v_r = -80.0
v_t = -30.0
eta = 0.0
a = 0.01
b = -20.0
d = 150.0
E_e = 0.0
E_i = -65.0
tau_s = 6.0
s_ext = 7.0*1e-3
v_spike = 40.0
v_reset = -55.0
theta_dist = "gaussian"
w_dist = "gaussian"

# define distribution of etas
f = gaussian if theta_dist == "gaussian" else lorentzian
thetas = f(N, mu=v_t, delta=Delta, lb=v_r, ub=2*v_t-v_r)

# define connectivity
indices = np.arange(0, N, dtype=np.int32)
pdfs = np.asarray([dist(idx, method=w_dist, zero_val=0.0, sigma=sigma) for idx in indices])
pdfs /= np.sum(pdfs)
W = circular_connectivity(N, p, spatial_distribution=rv_discrete(values=(indices, pdfs)), homogeneous_weights=False)

# define inputs
T = 3000.0
cutoff = 1000.0
dt = 1e-2
dts = 1e-1
inp = np.zeros((int(T/dt), N))
inp += poisson.rvs(mu=s_ext, size=inp.shape)
inp = convolve_exp(inp, tau_s, dt)

# run the model
###############

# initialize model
node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": thetas, "eta": eta, "tau_u": 1/a, "b": b, "kappa": d,
             "g_e": 0.0, "g_i": g, "E_e": E_e, "E_i": E_i, "tau_s": tau_s, "v": v_t}

# initialize model
net = Network(dt, device="cpu")
net.add_diffeq_node("ik", f"config/ik_snn/ik", weights=W, source_var="s", target_var="s_i",
                    input_var="g_e_in", output_var="s", spike_var="spike", reset_var="v", to_file=False,
                    node_vars=node_vars.copy(), op="ik_op", spike_reset=v_reset, spike_threshold=v_spike,
                    clear=True)

# perform simulation
obs = net.run(inputs=inp, sampling_steps=int(dts/dt), record_output=True, verbose=False, enable_grad=False,
              cutoff=int(cutoff/dt))
s = obs.to_numpy("out")

# calculate dimensionality
cov = s.T @ s
eigs = np.abs(np.linalg.eigvals(cov))
dim = np.sum(eigs) ** 2 / np.sum(eigs ** 2)
print(f"Dimensionality = {dim}")

# extract spikes in network
spike_counts = []
for idx in range(s.shape[1]):
    peaks, _ = find_peaks(s[:, idx])
    spike_counts.append(peaks)

    # fig, ax = plt.subplots(figsize=(12, 4))
    # ax.plot(s[:, idx]/np.max(s[:, idx]))
    # for p in peaks:
    #     ax.axvline(x=p, ymin=0.0, ymax=1.0, color="black", linestyle="dashed")
    # ax.set_xlabel("steps")
    # ax.set_ylabel("s")
    # plt.tight_layout()
    # plt.show()

# calculate sequentiality
print("Starting sequentiality calculation")
neighborhood = int(sigma*2)
overlap = 0.0
seq = sequentiality(spike_counts, neighborhood=neighborhood, overlap=overlap)
sqi = np.mean(seq)
print(f"Sequentiality = {sqi}")

# calculate firing rate statistics
taus = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0]
s_mean = np.mean(s, axis=1) / tau_s
s_std = np.std(s, axis=1) / tau_s
ffs, ffs2 = [], []
for tau in taus:
    ffs.append(fano_factor(spike_counts, s.shape[0], int(tau/dts)))
    ffs2.append(fano_factor2(spike_counts, s.shape[0], int(tau/dts)))

# save results
# pickle.dump({"g": g, "Delta": Delta, "theta_dist": theta_dist, "dim": dim, "s_mean": s_mean, "s_std": s_std,
#              "ff_between": ffs, "ff_within": ffs2, "ff_windows": taus},
#             open(f"/media/fsmresfiles/richard_data/numerics/dimensionality/spn_ss_g{int(g)}_D{int(Delta*10)}_{rep+1}.p",
#                  "wb"))

# plotting network connectivity
_, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(W, aspect="auto", interpolation="none", cmap="viridis")
plt.colorbar(im, ax=ax)
ax.set_xlabel("neuron")
ax.set_ylabel("neuron")
ax.set_title("W")
plt.tight_layout()

# plotting average firing rate dynamics
_, ax = plt.subplots(figsize=(12, 4))
ax.plot(s_mean*1e3, label="mean(r)")
ax.plot(s_std*1e3, label="std(r)")
ax.legend()
ax.set_xlabel("steps")
ax.set_ylabel("r")
ax.set_title(f"Dim = {dim}")
plt.tight_layout()

# plotting spiking dynamics
_, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(s.T, aspect="auto", interpolation="none", cmap="Greys")
plt.colorbar(im, ax=ax)
ax.set_xlabel("steps")
ax.set_ylabel("neurons")
plt.tight_layout()

# plotting fano factor distributions at different time scales
fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
for tau, ff1, ff2 in zip(taus, ffs, ffs2):
    ax = axes[0]
    ax.hist(ff1, label=f"tau = {tau}")
    ax.set_xlabel("ff")
    ax.set_ylabel("#")
    ax = axes[1]
    ax.hist(ff2, label=f"tau = {tau}")
    ax.set_xlabel("ff")
    ax.set_ylabel("#")
axes[0].set_title("time-specific FFs")
axes[0].legend()
axes[1].set_title("neuron-specific FFs")
axes[1].legend()
fig.suptitle("Fano Factor Distributions")
plt.tight_layout()

# plotting sequentiality distribution
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(seq)
ax.set_xlabel("sequentiality")
ax.set_ylabel("count")
ax.set_title(f"Sequentiality = {sqi}")
plt.tight_layout()

plt.show()
