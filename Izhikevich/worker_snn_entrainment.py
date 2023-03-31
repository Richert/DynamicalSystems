from rectipy import Network, circular_connectivity
import sys
# cond, wdir, tdir = sys.argv[-3:]
# sys.path.append(wdir)
sys.path.append("~/PycharmProjects/DynamicalSystems/reservoir_computing")
import numpy as np
from scipy.stats import cauchy
import matplotlib.pyplot as plt
import pickle
from scipy.stats import rv_discrete
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate, butter, sosfilt
from typing import Union


def butter_bandpass_filter(data: np.ndarray, freqs: Union[float, tuple], fs: int, order: int) -> np.ndarray:
    sos = butter(order, freqs, btype="low", output="sos", fs=fs)
    return sosfilt(sos, data, axis=-1)


def lorentzian(n: int, eta: float, delta: float, lb: float, ub: float):
    samples = np.zeros((n,))
    for i in range(n):
        s = cauchy.rvs(loc=eta, scale=delta)
        while s <= lb or s >= ub:
            s = cauchy.rvs(loc=eta, scale=delta)
        samples[i] = s
    return samples


def dist(x: int, method: str = "inverse", zero_val: float = 1.0, inverse_pow: float = 1.0) -> float:
    if method == "inverse":
        return 1/x**inverse_pow if x > 0 else zero_val
    if method == "exp":
        return np.exp(-x) if x > 0 else zero_val
    else:
        raise ValueError("Invalid method.")


def corr(s1: np.ndarray, s2: np.ndarray, method: str = "direct", max_lag: int = 100) -> np.ndarray:
    padding = list(np.zeros((max_lag,)))
    s2 = np.asarray(padding + list(s2) + padding)
    return np.max(correlate(s1, s2, mode="valid", method=method))


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
N = 2000
p = 0.2
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
Delta = 0.1
eta = 55.0
a = 0.03
b = -2.0
d = 100.0
g = 15.0
E_r = 0.0
tau_s = 6.0
v_spike = 1000.0
v_reset = -1000.0

# device for computations
device = "cuda:0"

# working directory
wdir = "config"
tdir = "results"

# sweep condition
cond = 105
p1 = "Delta"
p2 = "trial"

# parameter sweep definition
with open(f"{wdir}/entrainment_sweep.pkl", "rb") as f:
    sweep = pickle.load(f)
    v1s = sweep[p1]
    v2s = sweep[p2]
    f.close()
vals = [(v1, v2) for v1 in v1s for v2 in v2s]
v1, v2 = vals[int(cond)]
print(f"Condition: {p1} = {v1},  {p2} = {v2}")

# simulation-related parameters
T = 3000.0
dt = 1e-2
sr = 10
p_in = 0.1
omega = 0.005
alpha = 1e-2
steps = int(T/dt)
n_inputs = int(p_in*N)
center = int(N*0.5)
inp_indices = np.arange(center-int(0.5*n_inputs), center+int(0.5*n_inputs))
inp_dist = np.zeros((N,))
inp_dist[inp_indices] = 1.0
cutoff = int(1000.0/(dt*sr))

# define stimulation signal
time = np.linspace(0, T, steps)
driver_tmp = np.sin(2.0*np.pi*omega*time)
driver = np.zeros_like(driver_tmp)
driver[driver_tmp > 0.9] = 1.0

# downsample driver and calculate maximum correlation between driver and itself
driver_ds = driver[::sr]
max_corr = np.max(correlate(driver_ds, driver_ds, mode="valid"))

# infer stimulation onsets
margin = 200
driver_diff = np.diff(driver_ds[cutoff:])
stim_onsets = np.argwhere(driver_diff > 0.1).squeeze()
stim_width = np.argwhere(driver_diff < -0.1).squeeze()[0] - stim_onsets[0]
min_isi = np.min(np.abs(np.diff(stim_onsets))) - margin

# other analysis parameters
sigma = 10
f_high = 7.0
f_order = 5
f_cutoff = 2500
indices = np.arange(0, N, dtype=np.int32)
conn_pows = np.linspace(0.5, 1.5, num=3)

# define lorentzian of etas
thetas = lorentzian(N, eta=v_t, delta=Delta, lb=v_r, ub=0.0)

# simulation
############

# adjust parameters according to sweep condition
for param, v in zip([p1, p2], [v1, v2]):
    exec(f"{param} = {v}")

# prepare results storage
results = {"sweep": {p1: v1, p2: v2}, "T": T, "dt": dt, "sr": sr, "p": p, "alpha": alpha, "thetas": thetas,
           "input_indices": inp_indices, "dimensionality": [], "sequentiality": [], "corr_driven": [],
           "corr_nondriven": [], "population_avg": [], "population_var": [], "omegas": [], "conn_pows": conn_pows}
for i, conn_pow in enumerate(conn_pows):

    # define connectivity
    pdfs = np.asarray([dist(idx, method="inverse", zero_val=0.0, inverse_pow=conn_pow) for idx in indices])
    pdfs /= np.sum(pdfs)
    W = circular_connectivity(N, p, spatial_distribution=rv_discrete(values=(indices, pdfs)), homogeneous_weights=False)

    # initialize model
    node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": thetas, "eta": eta, "tau_u": 1/a, "b": b, "kappa": d,
                 "g": g, "E_r": E_r, "tau_s": tau_s, "v": v_t}

    # initialize model
    net = Network.from_yaml(f"{wdir}/ik_snn/rs", weights=W, source_var="s", target_var="s_in",
                            input_var="I_ext", output_var="s", spike_var="spike", spike_def="v", to_file=False,
                            node_vars=node_vars.copy(), op="rs_op", spike_reset=v_reset, spike_threshold=v_spike,
                            dt=dt, verbose=False, clear=True, device=device)

    # define input
    inp = np.zeros((steps, N))
    for idx in inp_indices:
        inp[:, idx] = driver*alpha

    # perform simulation
    obs = net.run(inputs=inp, sampling_steps=sr, record_output=True, verbose=False)
    res = obs["out"].iloc[cutoff:, :].values
    s = gaussian_filter1d(res, sigma=sigma, axis=0).T

    # calculate the correlation between the network
    snn_driven = np.mean(s[inp_indices, :], axis=0)
    snn_nondriven = np.mean(np.concatenate([s[:inp_indices[0], :], s[inp_indices[-1]:, :]], axis=0), axis=0)
    corr_driven = np.corrcoef(snn_driven, driver_ds[cutoff:])[0, 1]
    corr_nondriven = np.corrcoef(snn_nondriven, driver_ds[cutoff:])[0, 1]

    # calculate the network dimensionality
    corr_net = np.cov(s)
    eigs = np.abs(np.linalg.eigvals(corr_net))
    dim = np.sum(eigs)**2/np.sum(eigs**2)

    # calculate the network sequentiality
    sequentiality_measures = []
    for sidx in stim_onsets:
        if s.shape[1] - sidx > min_isi:
            s_tmp = s[:, sidx:sidx + min_isi]
            sequentiality_measures.append(sequentiality(s_tmp, neighborhood=50, max_lag=margin))

    # calculate the steady-state network activity
    population_dist = np.mean(s, axis=1).squeeze()
    population_dist /= np.max(population_dist)

    # calculate power in the bandpass-filtered signal
    r_filtered = butter_bandpass_filter(s, freqs=f_high, fs=int(1e3/(dt*sr)), order=f_order) * (1e3/tau_s)
    r_var = np.var(r_filtered[:, f_cutoff:-f_cutoff], axis=1)

    # store results
    results["corr_driven"].append(corr_driven)
    results["corr_nondriven"].append(corr_nondriven)
    results["dimensionality"].append(dim)
    results["sequentiality"].append(np.mean(sequentiality_measures))
    results["population_avg"].append(population_dist)
    results["population_var"].append(r_var)
    results["omegas"].append(omega)

    # plot results
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
    ax = axes[0, 0]
    im = ax.imshow(s, aspect="auto", interpolation="none")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xlabel('time')
    ax.set_ylabel('neurons')
    ax.set_title("Network activity raw")
    ax = axes[1, 0]
    ax.plot(population_dist)
    ax.set_xlabel("neuron id")
    ax.set_ylabel("s")
    ax.set_title("Mean activity over 2 s")
    ax = axes[0, 1]
    im = ax.imshow(r_filtered, aspect="auto", interpolation="none")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xlabel('time')
    ax.set_ylabel('neurons')
    ax.set_title(f"Network activity < {np.round(f_high, decimals=1)} Hz")
    ax = axes[1, 1]
    ax.plot(r_var)
    ax.set_xlabel("neuron id")
    ax.set_ylabel("var")
    ax.set_title(f"Signal variance < {np.round(f_high, decimals=1)} Hz")
    plt.tight_layout()
    plt.show()

# save results
fname = f"snn_entrainment"
with open(f"{tdir}/{fname}_{cond}.pkl", "wb") as f:
    pickle.dump(results, f)
    f.close()
