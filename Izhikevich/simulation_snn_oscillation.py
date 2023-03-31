from rectipy import Network, circular_connectivity, line_connectivity
import sys
sys.path.append("~/PycharmProjects/DynamicalSystems/reservoir_computing")
import numpy as np
from reservoir_computing.utility_funcs import lorentzian, dist
import matplotlib.pyplot as plt
import pickle
from scipy.stats import rv_discrete
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import dice


def corr(s1: np.ndarray, s2: np.ndarray, max_lag: int = 100, normalize: bool = True, threshold: float = 1e-6
         ) -> np.ndarray:
    max_shift = int(2 * max_lag)
    if normalize:
        s2_max = np.max(s2)
        if s2_max < threshold:
            return np.zeros((max_shift,))
        s2 = s2 / np.max(s2)
    padding = list(np.zeros((max_lag,)))
    s2 = np.asarray(padding + list(s2) + padding)
    corrs = []
    max_len = len(s2)
    for lag in range(max_shift):
        corrs.append(np.corrcoef(s1, s2[lag:max_len-max_shift+lag])[0, 1])
    return np.asarray(corrs)


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
thetas = lorentzian(N, eta=v_t, delta=Delta, lb=v_r, ub=0.0)

# define connectivity
indices = np.arange(0, N, dtype=np.int32)
pdfs = np.asarray([dist(idx, method="inverse", zero_val=0.0, inverse_pow=1.5) for idx in indices])
pdfs /= np.sum(pdfs)
W = circular_connectivity(N, p, spatial_distribution=rv_discrete(values=(indices, pdfs)), homogeneous_weights=False)
# plt.imshow(W, interpolation="none", aspect="equal")
# plt.show()
print(np.sum(np.sum(W, axis=1)))

# define inputs
cutoff = 1000.0
T = 3000.0
dt = 1e-2
sr = 10
p_in = 0.1
omega = 0.005
steps = int(T/dt)
inp = np.zeros((steps, N))
time = np.linspace(0, T, steps)
driver = np.sin(2.0*np.pi*omega*time)
for idx in range(int(N*p_in)):
    inp[driver > 0.9, idx] = 1e-2

# run the model
###############

# initialize model
node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": thetas, "eta": eta, "tau_u": 1/a, "b": b, "kappa": d,
             "g": g, "E_r": E_r, "tau_s": tau_s, "v": v_t}

# initialize model
net = Network.from_yaml(f"config/ik_snn/rs", weights=W, source_var="s", target_var="s_in",
                        input_var="s_ext", output_var="s", spike_var="spike", spike_def="v", to_file=False,
                        node_vars=node_vars.copy(), op="rs_op", spike_reset=v_reset, spike_threshold=v_spike,
                        dt=dt, verbose=False, clear=True, device="cuda:0")

# perform simulation
obs = net.run(inputs=inp, sampling_steps=sr, record_output=True, verbose=False)
res = obs["out"]
s = gaussian_filter1d(res.values, sigma=10, axis=0)

# infer stimulation onsets
margin = 200
start = int(cutoff/(dt*sr))
driver_signal = np.zeros_like(driver[::sr])
driver_signal[driver[::sr] > 0.9] = 1.0
driver_diff = np.diff(driver_signal)
stim_onsets = np.argwhere(driver_diff > 0.1).squeeze()
stim_onsets = stim_onsets[stim_onsets > start]
stim_width = np.argwhere(driver_diff < -0.1).squeeze()[0] - stim_onsets[0]
min_isi = np.min(np.abs(np.diff(stim_onsets))) - margin

# calculate correlation between driver and driven network
snn_driven = np.mean(s[:, :int(N*p_in)], axis=1)
correlation = np.corrcoef(snn_driven[start:], driver_signal[start:])[0, 1]

# calculate sequentiality of network dynamics
sequentiality_measures = []
for sidx in stim_onsets:
    s_tmp = s[sidx:sidx+min_isi, :]
    sequentiality_measures.append(sequentiality(s_tmp.T, neighborhood=50, max_lag=margin))

# plot results
fig, axes = plt.subplots(nrows=2, figsize=(12, 9))
ax = axes[0]
ax.plot(driver_signal, label="input")
ax.plot(np.mean(s, axis=1), label="mean-field")
ax.plot(np.mean(s[:, :int(N*p_in)], axis=1), label="driven")
ax.plot(np.mean(s[:, int(N*p_in):], axis=1), label="non-driven")
ax.legend()
ax.set_xlabel("time")
ax.set_ylabel("s")
ax.set_title(f"Input-Driven Correlation: {correlation}")
ax = axes[1]
im = ax.imshow(s.T, aspect="auto", interpolation="none")
plt.colorbar(im, ax=ax, shrink=0.4)
ax.set_xlabel('time')
ax.set_ylabel('neurons')
ax.set_title(f"Sequentiality of SNN dynamics = {np.mean(sequentiality_measures)} ")
plt.tight_layout()

# saving
fig.canvas.draw()
plt.savefig(f'results/snn_oscillations_het.pdf')
plt.show()
