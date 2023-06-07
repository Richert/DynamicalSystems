from rectipy import Network, circular_connectivity
import sys
sys.path.append("~/PycharmProjects/DynamicalSystems/reservoir_computing")
import numpy as np
from scipy.stats import cauchy
import matplotlib.pyplot as plt
import pickle
from scipy.stats import rv_discrete
from scipy.ndimage import gaussian_filter1d
import h5py


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


def get_signals(stim_onsets: list, cycle_steps: int, sr: int, net: Network, y0: dict, inp_indices: np.ndarray,
                sigma: float = 10.0):

    # perform simulation for each stimulation time
    signals = []
    inputs = []
    start = int(np.round(cycle_steps / sr))
    for i, stim in enumerate(stim_onsets):
        inp = np.zeros((stim + cycle_steps, N))
        inp[stim:stim + stim_width, inp_indices] = alpha
        inp = gaussian_filter1d(inp, sigma=sigma, axis=0)
        net.reset(y0)
        obs = net.run(inputs=inp, sampling_steps=sr, record_output=True, verbose=False, enable_grad=False)
        res = obs.to_numpy("out")[-start:, :]
        signals.append(gaussian_filter1d(res, sigma=sigma, axis=0).T)
        inputs.append(inp[::sr, inp_indices[0]][-start:])

    return signals, inputs


def mse(x: np.ndarray, y: np.ndarray) -> float:
    x = x.squeeze() / np.max(x.flatten())
    y = y.squeeze() / np.max(y.flatten())
    return float(np.mean((x - y)**2))


def get_c(X: np.ndarray, alpha: float = 1e-4):
    """
    """
    return X @ X.T + alpha*np.eye(X.shape[0])


def pca(X: np.ndarray) -> tuple:
    X = X - X.mean()
    Z = X / X.std()
    C = Z.T @ Z
    eigvals, eigvecs = np.linalg.eig(C)
    if np.abs(np.min(eigvecs[:, 0])) > np.abs(np.max(eigvecs[:, 0])):
        eigvecs[:, 0] *= -1
    vals_abs = np.abs(eigvals)
    sort_idx = np.argsort(vals_abs)[::-1]
    return eigvals[sort_idx]/np.sum(vals_abs), eigvecs[:, sort_idx]


# define parameters
###################

# condition
cond = "het"
alpha = 45.0
Delta = 1.0

# training and testing
n_stims = 20
n_tests = 5

# working directory
wdir = "config"
tdir = "results/oscillatory"

# load data that maps deltas to frequencies
data = pickle.load(open(f"{wdir}/fre_oscillations.pkl", "rb"))
deltas = data["deltas"]
freqs = data["freqs"]

# model parameters
N = 1000
p = 0.2
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
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

# define lorentzian of etas
thetas = lorentzian(N, eta=v_t, delta=Delta, lb=v_r, ub=2 * v_t - v_r)

# simulation parameters
T_init = 2000.0
dt = 1e-2
sr = 10
p_in = 0.2
idx = np.argmin(np.abs(deltas - Delta))
freq = freqs[idx]
T = 1e3/freq
cycle_steps = int(T/dt)
stim_onsets = np.linspace(0, T/3, num=n_stims+1)[:-1]
stim_onsets = [int(onset/dt) for onset in stim_onsets]
stim_width = int(20.0/dt)
n_inputs = int(p_in*N)
center = int(N*0.5)
inp_indices = np.arange(center-int(0.5*n_inputs), center+int(0.5*n_inputs))

# create two target signals to fit
delay = 2000
steps = int(np.round(cycle_steps / sr))
target_1 = np.zeros((steps,))
target_1[delay] = 1.0
target_1 = gaussian_filter1d(target_1, sigma=int(delay*0.1))
t = np.linspace(0, T*1e-3, steps)
f1 = 5.0
f2 = 20.0
target_2 = np.sin(2.0*np.pi*f1*t) * np.sin(2.0*np.pi*f2*t)
targets = [target_1, target_2]

# other analysis parameters
K_width = 100
sigma = 10
margin = 100
seq_range = 50
indices = np.arange(0, N, dtype=np.int32)
conn_pow = 0.75

# initial simulation
####################

# define connectivity
pdfs = np.asarray([dist(idx, method="inverse", zero_val=0.0, inverse_pow=conn_pow) for idx in indices])
pdfs /= np.sum(pdfs)
W = circular_connectivity(N, p, spatial_distribution=rv_discrete(values=(indices, pdfs)), homogeneous_weights=False)

# collect parameters
node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": thetas, "eta": eta, "tau_u": 1/a, "b": b, "kappa": d,
             "g": g, "E_r": E_r, "tau_s": tau_s, "v": v_t}

# initialize model
net = Network(dt=dt, device="cuda:0")
net.add_diffeq_node("rs", node=f"{wdir}/ik_snn/rs", weights=W, source_var="s", target_var="s_in",
                    input_var="I_ext", output_var="s", spike_var="spike", spike_def="v", to_file=False,
                    node_vars=node_vars.copy(), op="rs_op", spike_reset=v_reset, spike_threshold=v_spike,
                    verbose=False, clear=True)

# perform initial wash-out simulation
init_steps = int(T_init/dt)
inp = np.zeros((init_steps, 1))
net.run(inputs=inp, sampling_steps=init_steps, verbose=False, enable_grad=False)
y0 = net.state

# main simulation
#################

fname = f"SI_oscillations_{cond}"
f = f"{tdir}/{fname}.h5"
hf = h5py.File(f, "w")
g = hf.create_group("sweep")
for key, val in {"Delta": Delta, "alpha": alpha}.items():
    g.create_dataset(key, data=val)
hf.close()

# define connectivity
pdfs = np.asarray([dist(idx, method="inverse", zero_val=0.0, inverse_pow=conn_pow) for idx in indices])
pdfs /= np.sum(pdfs)
W = circular_connectivity(N, p, spatial_distribution=rv_discrete(values=(indices, pdfs)), homogeneous_weights=False)

# initialize model
net = Network(dt=dt, device="cuda:0")
net.add_diffeq_node("rs", node=f"{wdir}/ik_snn/rs", weights=W, source_var="s", target_var="s_in",
                    input_var="I_ext", output_var="s", spike_var="spike", spike_def="v", to_file=False,
                    node_vars=node_vars.copy(), op="rs_op", spike_reset=v_reset, spike_threshold=v_spike,
                    verbose=False, clear=True)

# perform initial wash-out simulation
init_steps = int(T_init / dt)
inp = np.zeros((init_steps, 1))
net.run(inputs=inp, sampling_steps=init_steps, verbose=False, enable_grad=False)
y0 = net.state

# get signals for each stimulation onset
signals, inputs = get_signals(stim_onsets, cycle_steps, sr, net, y0, inp_indices, sigma=sigma)

dims = []
seqs = []
cs = []
for s in signals:

    # calculate the network dimensionality
    corr_net = np.cov(s)
    eigs = np.abs(np.linalg.eigvals(corr_net))
    dims.append(np.sum(eigs) ** 2 / np.sum(eigs ** 2))

    # calculate the network sequentiality
    seqs.append(sequentiality(s, neighborhood=seq_range, max_lag=margin))

    # calculate the network covariance matrices
    cs.append(get_c(s, alpha=1e-4))

# calculate the network kernel
s_mean = np.mean(signals, axis=0)
s_var = np.mean([s_i - s_mean for s_i in signals], axis=0)
C_inv = np.linalg.inv(np.mean(cs, axis=0))
C_inv2 = np.linalg.inv(s_mean @ s_mean.T + alpha*np.eye(s_mean.shape[0]) + 1/n_stims * s_var @ s_var.T)
w1 = C_inv @ s_mean
w2 = C_inv2 @ s_mean
K1 = s_mean.T @ w1
G1 = s_var.T @ w1
K2 = s_mean.T @ w2
G2 = s_var.T @ w2

# calculate the network response on test data
test_onsets = np.random.randint(low=np.min(stim_onsets), high=np.max(stim_onsets), size=n_tests)
test_signals, _ = get_signals(list(test_onsets), cycle_steps, sr, net, y0, inp_indices, sigma=sigma)

# calculate the prediction performance for concrete targets
train_predictions = []
train_distortions = []
test_predictions = []
readouts = []
for target in targets:
    train_predictions.append(K2 @ target)
    train_distortions.append(G2 @ target)
    w_readout = w2 @ target
    readouts.append(w_readout)
    test_predictions.append([w_readout @ test_sig for test_sig in test_signals])

# calculate the network kernel basis functions
# K_funcs = np.zeros((K.shape[0] - 2 * K_width, K_width))
# for j in range(K_funcs.shape[0]):
#     rows = np.arange(K_width + j, 2 * K_width + j)
#     cols = rows[::-1]
#     K_funcs[j, :] = K[rows, cols]
K_shifted = np.zeros_like(K2)
for j in range(K2.shape[0]):
    K_shifted[j, :] = np.roll(K2[j, :], shift=int(K2.shape[1]/2)-j)
v_explained, pcs = pca(K_shifted)
pc1_proj = K_shifted @ pcs[:, 0]

# store results
hf = h5py.File(f, 'r+')
g = hf.create_group(f"data")
results = {"T": T, "dt": dt, "sr": sr, "p": p, "thetas": thetas,
           "input_indices": inp_indices, "dimensionality": np.mean(dims), "sequentiality": np.mean(seqs),
           "alpha": alpha, "train_predictions": train_predictions, "targets": targets,
           "distortions": train_distortions, "test_predictions": test_predictions,
           "test_onsets": np.round(dt * 2.0 * np.pi * test_onsets / T, decimals=2),
           "v_explained": v_explained, "pc1_projection": pc1_proj, "pcs": pcs, "s": test_signals, "K": K2,
           "K_shifted": K_shifted, }
for key, val in results.items():
    g.create_dataset(key, data=val)
hf.close()

# plot results
_, axes = plt.subplots(nrows=2, figsize=(12, 5))
s_all = np.concatenate(signals, axis=1)
inp_all = np.concatenate(inputs, axis=0)
s_all /= np.max(s_all)
inp_all /= np.max(inp_all)
ax = axes[0]
ax.plot(np.mean(s_all, axis=0), label="s")
ax.plot(inp_all, label="I_ext")
ax.legend()
ax.set_xlabel("time")
ax.set_title("Mean signal")
ax = axes[1]
im = ax.imshow(s_all, aspect="auto", interpolation="none")
plt.colorbar(im, ax=ax, shrink=0.4)
ax.set_xlabel('time')
ax.set_ylabel('neurons')
ax.set_title(f"Seq = {np.mean(seqs)}, Dim = {np.mean(dims)}")
plt.tight_layout()

_, axes = plt.subplots(ncols=2, figsize=(12, 6))
ax = axes[0]
ax.plot(target_1, label="target")
ax.plot(train_predictions[0], label="prediction")
ax.set_xlabel("time")
ax.set_title(f"T1")
ax.legend()
ax = axes[1]
ax.plot(target_2, label="target")
ax.plot(train_predictions[1], label="prediction")
ax.set_xlabel("time")
ax.set_title(f"T2")
ax.legend()
plt.tight_layout()

examples = [(0, 0), (0, 1), (1, 0), (1, 1)]
fig, axes = plt.subplots(nrows=len(examples), figsize=(12, 6))
for i, ex in enumerate(examples):
    ax = axes[i]
    ax.plot(test_predictions[ex[0]][ex[1]], label="prediction")
    ax.plot(targets[ex[0]], label="target")
    ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel("s")
    ax.set_title(f"Stimulation phase: {test_onsets[ex[1]]}")
plt.tight_layout()

fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
ax = axes[0]
im = ax.imshow(K1, interpolation="none", aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("K")
ax = axes[1]
im = ax.imshow(K2, interpolation="none", aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("K2")
ax = axes[2]
im = ax.imshow(C_inv - C_inv2, interpolation="none", aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("K - K2")

# saving
plt.show()
