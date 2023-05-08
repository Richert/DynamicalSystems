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


def get_signals(stim_onsets: list, cycle_steps: int, sr: int, net: Network, y0: dict, inp_indices: np.ndarray,
                sigma: float = 10.0):

    # perform simulation for each stimulation time
    signals = []
    inputs = []
    print("Simulating stimulation trials...")
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
        print(f"Trials finished: {(i + 1)} / {len(stim_onsets)}")

    return signals, inputs


def mse(x: np.ndarray, y: np.ndarray) -> float:
    x = x.squeeze() / np.max(x.flatten())
    y = y.squeeze() / np.max(y.flatten())
    return float(np.mean((x - y)**2))


# define parameters
###################

# load data that maps deltas to frequencies
data = pickle.load(open("config/fre_oscillations.pkl", "rb"))
deltas = data["deltas"]
freqs = data["freqs"]

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

# simulation parameters
T_init = 1000.0
dt = 1e-2
sr = 10
alpha = 40.0
p_in = 0.2
idx = np.argmin(np.abs(deltas - Delta))
freq = freqs[idx]
T = 1e3/freq
cycle_steps = int(T/dt)
stim_onsets = np.linspace(0, T/3, num=21)[:-1]
stim_onsets = [int(onset/dt) for onset in stim_onsets]
stim_width = int(20.0/dt)
n_inputs = int(p_in*N)
center = int(N*0.5)
inp_indices = np.arange(center-int(0.5*n_inputs), center+int(0.5*n_inputs))

# other analysis parameters
sigma = 10
margin = 100
seq_range = 50
spike_height = 0.7
spike_width = 50
isi_bins = 20
isi_min = 1500
indices = np.arange(0, N, dtype=np.int32)
conn_pow = 0.75

# simulations
#############

# define connectivity
pdfs = np.asarray([dist(idx, method="inverse", zero_val=0.0, inverse_pow=conn_pow) for idx in indices])
pdfs /= np.sum(pdfs)
W = circular_connectivity(N, p, spatial_distribution=rv_discrete(values=(indices, pdfs)), homogeneous_weights=False)
print(np.sum(np.sum(W, axis=1)))

# collect parameters
node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": thetas, "eta": eta, "tau_u": 1/a, "b": b, "kappa": d,
             "g": g, "E_r": E_r, "tau_s": tau_s, "v": v_t}

# initialize model
net = Network(dt=dt, device="cuda:0")
net.add_diffeq_node("rs", node=f"config/ik_snn/rs", weights=W, source_var="s", target_var="s_in",
                    input_var="I_ext", output_var="s", spike_var="spike", spike_def="v", to_file=False,
                    node_vars=node_vars.copy(), op="rs_op", spike_reset=v_reset, spike_threshold=v_spike,
                    verbose=False, clear=True)

# perform initial wash-out simulation
init_steps = int(T_init/dt)
inp = np.zeros((init_steps, 1))
net.run(inputs=inp, sampling_steps=init_steps, verbose=False, enable_grad=False)
y0 = net.state

# get signals for each stimulation onset
signals, inputs = get_signals(stim_onsets, cycle_steps, sr, net, y0, inp_indices, sigma=sigma)

# calculate the network dimensionality
print("Starting dimensionality calculation...")
dims = []
for i, s in enumerate(signals):
    corr_net = np.cov(s)
    eigs = np.abs(np.linalg.eigvals(corr_net))
    dims.append(np.sum(eigs)**2/np.sum(eigs**2))
    print(f"Trials finished: {(i + 1)} / {len(signals)}")

# calculate the network sequentiality
print("Starting sequentiality calculation...")
seqs = []
for i, s in enumerate(signals):
    seqs.append(sequentiality(s, neighborhood=seq_range, max_lag=margin))
    print(f"Trials finished: {(i + 1)} / {len(signals)}")

# calculate the network covariance matrices
print("Starting network covariance calculation...")
cs = []
for i, s in enumerate(signals):
    cs.append(get_c(s, alpha=1e-4))
    print(f"Trials finished: {(i + 1)} / {len(signals)}")

# calculate the network kernel
print("Starting network kernel calculation...")
s_mean = np.mean(signals, axis=0)
s_var = np.mean([s_i - s_mean for s_i in signals], axis=0)
C_inv = np.linalg.inv(np.mean(cs, axis=0))
w = C_inv @ s_mean
K = s_mean.T @ w
G = s_var.T @ w
print("Finished.")

# calculate the prediction performance for a concrete target
omega = 2000
sigma_t = int(omega * 0.1)
target = np.zeros((K.shape[0]))
target[omega] = alpha
# target = np.sin(2*np.pi*np.arange(0, int(np.round(cycle_steps / sr)))*omega*dt*sr*1e-3)
target = gaussian_filter1d(target, sigma=sigma_t)
prediction = K @ target
distortion = G @ target

# calculate the readout weights
w_readout = w @ target
w_magnitudes = np.argsort(w_readout)

# calculate the network kernel basis functions
print("Starting network basis function calculation...")
width = 100
K_funcs = np.zeros((K.shape[0] - 2*width, width))
for i in range(K_funcs.shape[0]):
    rows = np.arange(width+i, 2*width+i)
    cols = rows[::-1]
    K_funcs[i, :] = K[rows, cols]
v_explained, pcs = pca(K_funcs)
pc1_proj = K_funcs @ pcs[:, 0]
print("Finished.")

# calculate the network response on test data
test_onsets = np.random.randint(low=np.min(stim_onsets), high=np.max(stim_onsets), size=5)
test_signals, test_inputs = get_signals(list(test_onsets), cycle_steps, sr, net, y0, inp_indices, sigma=sigma)
test_loss = np.mean([mse(test_sig, target) for test_sig in test_signals])
print(f"Test loss (MSE between target and prediction) = {test_loss}")

# plotting
##########

fig, axes = plt.subplots(nrows=4, figsize=(12, 9))
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
im = ax.imshow(s_all[w_magnitudes, :], aspect="auto", interpolation="none")
plt.colorbar(im, ax=ax, shrink=0.4)
ax.set_xlabel('time')
ax.set_ylabel('neurons')
ax.set_title(f"Seq = {np.mean(seqs)}, Dim = {np.mean(dims)}")
ax = axes[2]
mse = np.mean((K - np.eye(K.shape[0]))**2, axis=0)
ax.plot(target, label="target")
ax.plot(prediction, label="prediction")
ax.set_xlabel("lags")
ax.set_ylabel("Mean reconstruction")
ax.legend()
ax.set_title(f"Mean MSE: {np.mean(mse)}")
ax = axes[3]
ax.plot(distortion)
ax.set_xlabel("lags")
ax.set_ylabel("mean(G[:, lag])")
ax.set_title(f"Mean G: {np.mean(G.flatten())}")
plt.tight_layout()

# plot kernel
fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
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

# plot readout
examples = [0, 1, 2]
fig, axes = plt.subplots(nrows=len(examples), figsize=(12, 6))
for i, ex in enumerate(examples):
    ax = axes[i]
    ax.plot(w_readout @ test_signals[ex], label="prediction")
    ax.plot(target, label="target")
    ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel("s")
    ax.set_title(f"Stimulation phase: {np.round(dt*2.0*np.pi*test_onsets[ex]/T, decimals=2)}")
plt.tight_layout()

# plot PCA results
fig, axes = plt.subplots(nrows=4, figsize=(12, 8))
ax = axes[0]
ax.plot(pcs[:, 0])
ax.set_xlabel("lag")
ax.set_ylabel("magnitude")
ax.set_title("PC1")
ax = axes[1]
ax.plot(pcs[:, 1])
ax.set_xlabel("lag")
ax.set_ylabel("magnitude")
ax.set_title("PC2")
ax = axes[2]
ax.plot(v_explained)
ax.set_xlabel("PC")
ax.set_ylabel("ratio")
ax.set_title("Variance explained by PCs")
ax = axes[3]
ax.plot(pc1_proj)
ax.set_xlabel("time")
ax.set_ylabel("weight")
ax.set_title("Kernel projection onto PC1")
plt.tight_layout()

# saving
plt.show()
