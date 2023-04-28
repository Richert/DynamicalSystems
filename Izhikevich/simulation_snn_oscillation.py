from rectipy import Network, circular_connectivity
import sys
sys.path.append("~/PycharmProjects/DynamicalSystems/reservoir_computing")
import numpy as np
from reservoir_computing.utility_funcs import lorentzian, dist
import matplotlib.pyplot as plt
import pickle
from scipy.stats import rv_discrete
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def _sequentiality(signals: np.ndarray, max_lag: int, threshold: float = 1e-6) -> tuple:
    N = signals.shape[0]
    sym = 0
    asym = 0
    padding = list(np.zeros((max_lag,)))
    lags = np.arange(1, max_lag+1)
    for n1 in range(N):
        s1 = np.asarray(padding + list(signals[n1]) + padding)
        for n2 in range(N):
            s2 = signals[n2]
            if np.max(s1) > threshold and np.max(s2) > threshold:
                cc = np.correlate(s1, s2, mode="valid")
                cc_pos = cc[max_lag+lags]
                cc_neg = cc[max_lag-lags]
                sym += np.sum((cc_pos - cc_neg) ** 2)
                asym += np.sum((cc_pos + cc_neg) ** 2)
    return sym, asym


def sequentiality(signals: np.ndarray, max_lag: int, neighborhood: int, overlap: float = 0.5) -> float:
    N = signals.shape[0]
    sym, asym = [], []
    idx = 0
    while idx < N:

        # select part of network that we will calculate the sequentiality for
        start = np.maximum(0, idx-int(overlap*neighborhood))
        stop = np.minimum(N, start+neighborhood)
        idx = stop

        # calculate sequentiality
        res = _sequentiality(signals[start:stop], max_lag=max_lag)
        if res[1] > 1e-8:
            sym.append(res[0])
            asym.append(res[1])

    return np.mean([np.sqrt(s/a) for s, a in zip(sym, asym)]).squeeze()


def get_cinv(X: np.ndarray, alpha: float = 1e-4):
    """
    """
    return np.linalg.inv(X @ X.T + alpha*np.eye(X.shape[0]))


# define parameters
###################

# model parameters
N = 1000
p = 0.2
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
Delta = 1.0
eta = 55.0
a = 0.03
b = -2.0
d = 100.0
g = 15.0
E_r = 0.0
tau_s = 6.0
v_spike = 1000.0
v_reset = -1000.0

# define lorentzian of etas
thetas = lorentzian(N, eta=v_t, delta=Delta, lb=v_r, ub=2 * v_t - v_r)

# simulation-related parameters
T = 3000.0
dt = 1e-2
sr = 10
p_in = 0.25
alpha = 10.0
omega = 5.1
steps = int(T/dt)
n_inputs = int(p_in*N)
center = int(N*0.5)
inp_indices = np.arange(center-int(0.5*n_inputs), center+int(0.5*n_inputs))
cutoff = int(1000.0/(dt*sr))

# define stimulation signal
time = np.linspace(0, T, steps)
driver_tmp = np.sin(2.0 * np.pi * omega * 1e-3 * time)
driver = np.zeros_like(driver_tmp)
driver[driver_tmp > 0.9] = 1.0
driver_ds = driver[::sr]

# infer stimulation onsets
margin = 200
driver_diff = np.diff(driver_ds[cutoff:])
stim_onsets = np.argwhere(driver_diff > 0.1).squeeze()
stim_width = np.argwhere(driver_diff < -0.1).squeeze()[0] - stim_onsets[0]
min_isi = np.min(np.abs(np.diff(stim_onsets))) - margin

# other analysis parameters
sigma = 10
spike_height = 0.7
spike_width = 50
isi_bins = 20
isi_min = 1500
indices = np.arange(0, N, dtype=np.int32)
conn_pow = 0.75

# run the model
###############

# define connectivity
pdfs = np.asarray([dist(idx, method="inverse", zero_val=0.0, inverse_pow=conn_pow) for idx in indices])
pdfs /= np.sum(pdfs)
W = circular_connectivity(N, p, spatial_distribution=rv_discrete(values=(indices, pdfs)), homogeneous_weights=False)
print(np.sum(np.sum(W, axis=1)))

# initialize model
node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": thetas, "eta": eta, "tau_u": 1/a, "b": b, "kappa": d,
             "g": g, "E_r": E_r, "tau_s": tau_s, "v": v_t}

# initialize model
net = Network(dt=dt, device="cuda:0")
net.add_diffeq_node("rs", node=f"config/ik_snn/rs", weights=W, source_var="s", target_var="s_in",
                    input_var="I_ext", output_var="s", spike_var="spike", spike_def="v", to_file=False,
                    node_vars=node_vars.copy(), op="rs_op", spike_reset=v_reset, spike_threshold=v_spike,
                    verbose=False, clear=True)

# define input
inp = np.zeros((steps, N))
for idx in inp_indices:
    inp[:, idx] = driver*alpha

# perform simulation
obs = net.run(inputs=inp, sampling_steps=sr, record_output=True, verbose=False)
res = obs.to_numpy("out")
s = gaussian_filter1d(res[cutoff:, :], sigma=sigma, axis=0).T

# calculate the correlation between the network and the driver
s_smoothed = gaussian_filter1d(res, sigma=20*sigma, axis=0).T
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(s_smoothed, aspect="auto", interpolation="none")
plt.show()
freqs = []
for idx in range(s_smoothed.shape[0]):
    s_tmp = s_smoothed[idx, :]
    if np.max(s_tmp) > 1e-12:
        s_tmp /= np.max(s_tmp)
        peaks, _ = find_peaks(s_tmp[cutoff:], prominence=0.5, width=200.0)
    else:
        peaks = []
    if len(peaks) > 1:
        isis = np.diff(peaks)
        freqs.append(1e3/(np.mean(isis)*dt*sr))
    else:
        freqs.append(0.0)
snn_driven = np.mean(s[inp_indices, :], axis=0)
snn_nondriven = np.mean(np.concatenate([s[:inp_indices[0], :], s[inp_indices[-1]:, :]], axis=0), axis=0)
corr_driven = np.corrcoef(snn_driven, driver_ds[cutoff:])[0, 1]
corr_nondriven = np.corrcoef(snn_nondriven, driver_ds[cutoff:])[0, 1]

# calculate the network dimensionality
corr_net = np.cov(s)
eigs = np.abs(np.linalg.eigvals(corr_net))
dim = np.sum(eigs)**2/np.sum(eigs**2)

# calculate the network sequentiality and kernel rank
sequentiality_measures = []
signals = []
c_inverses = []
for sidx in stim_onsets:
    if s.shape[1] - sidx > min_isi:
        s_tmp = s[:, sidx:sidx + min_isi]
        sequentiality_measures.append(sequentiality(s_tmp, neighborhood=50, max_lag=margin))
        signals.append(s_tmp)
        c_inverses.append(get_cinv(s_tmp))
s_mean = np.mean(signals, axis=0)
s_var = np.mean([s_i - s_mean for s_i in signals], axis=0)
C_inv = np.mean(c_inverses, axis=0)
K = s_mean.T @ C_inv @ s_mean
G = s_var.T @ C_inv @ s_mean

# plot results
fig, axes = plt.subplots(nrows=4, figsize=(12, 9))
ax = axes[0]
ax.plot(driver_ds[cutoff:], label="input")
ax.plot(np.mean(s, axis=0), label="mean-field")
ax.plot(snn_driven, label="driven")
ax.plot(snn_nondriven, label="non-driven")
ax.legend()
ax.set_xlabel("time")
ax.set_ylabel("s")
ax.set_title(f"Corr_driven = {corr_driven}, corr_nondriven = {corr_nondriven}")
ax = axes[1]
im = ax.imshow(s, aspect="auto", interpolation="none")
plt.colorbar(im, ax=ax, shrink=0.4)
ax.set_xlabel('time')
ax.set_ylabel('neurons')
ax.set_title(f"Seq = {np.mean(sequentiality_measures)}, Dim = {dim}")
ax = axes[2]
ax.plot(np.mean((K - np.eye(K.shape[0]))**2, axis=0))
ax.set_xlabel("lags")
ax.set_ylabel("MSE(K-I)")
ax = axes[3]
ax.plot(np.mean(G, axis=0))
ax.set_xlabel("lags")
ax.set_ylabel("mean(G[:, lag])")
plt.tight_layout()

_, axes = plt.subplots(ncols=2, figsize=(12, 6))
ax = axes[0]
ax.imshow(K, aspect="auto", interpolation="none")
ax.set_xlabel("lags")
ax.set_ylabel("lags")
ax.set_title(f"K")
ax = axes[1]
ax.imshow(G, aspect="auto", interpolation="none")
ax.set_xlabel("lags")
ax.set_ylabel("lags")
ax.set_title(f"G")
plt.tight_layout()

# saving
fig.canvas.draw()
# plt.savefig(f'results/snn_oscillations_het.pdf')
plt.show()
