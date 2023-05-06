from rectipy import Network, circular_connectivity
import sys
cond, wdir, tdir = sys.argv[-3:]
sys.path.append(wdir)
sys.path.append("~/PycharmProjects/DynamicalSystems/reservoir_computing")
import numpy as np
from scipy.stats import cauchy
import matplotlib.pyplot as plt
import pickle
from scipy.stats import rv_discrete
from scipy.ndimage import gaussian_filter1d


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

# load data that maps deltas to frequencies
data = pickle.load(open("results/fre_oscillations.pkl", "rb"))
deltas = data["deltas"]
freqs = data["freqs"]

# model parameters
N = 2000
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

# device for computations
device = "cuda:0"

# working directory
# wdir = "config"
# tdir = "results"

# sweep condition
# cond = 10
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
omegas = sweep["omega"]
omega = omegas[np.argmin(np.abs(v1s - v1))]
print(f"Condition: {p1} = {v1},  {p2} = {v2}, omega = {omega}")

# adjust parameters according to sweep condition
for param, v in zip([p1, p2], [v1, v2]):
    exec(f"{param} = {v}")

# define lorentzian of etas
thetas = lorentzian(N, eta=v_t, delta=Delta, lb=v_r, ub=2 * v_t - v_r)

# simulation parameters
T_init = 2000.0
dt = 1e-2
sr = 10
alphas = np.linspace(1, 100, 20)
p_in = 0.2
idx = np.argmin(np.abs(deltas - Delta))
freq = freqs[idx]
T = 1e3/freq
cycle_steps = int(T/dt)
n_stims = 20
n_tests = 5
stim_onsets = np.linspace(0, T/3, num=n_stims+1)[:-1]
stim_onsets = [int(onset/dt) for onset in stim_onsets]
stim_width = int(20.0/dt)
n_inputs = int(p_in*N)
center = int(N*0.5)
inp_indices = np.arange(center-int(0.5*n_inputs), center+int(0.5*n_inputs))

# other analysis parameters
sigma = 10
margin = 100
seq_range = 50
indices = np.arange(0, N, dtype=np.int32)
conn_pow = 0.75

# simulations
#############

# define connectivity
pdfs = np.asarray([dist(idx, method="inverse", zero_val=0.0, inverse_pow=conn_pow) for idx in indices])
pdfs /= np.sum(pdfs)
W = circular_connectivity(N, p, spatial_distribution=rv_discrete(values=(indices, pdfs)), homogeneous_weights=False)

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

# simulation
############

# prepare results storage
results = {"sweep": {p1: v1, p2: v2}, "T": T, "dt": dt, "sr": sr, "p": p, "thetas": thetas,
           "input_indices": inp_indices, "dimensionality": [], "sequentiality": [], "Ks": [], "Gs": [],
           "test_signals": [], "test_onsets": [], "omegas": omegas, "alphas": alphas}
for i, alpha in enumerate(alphas):

    # define connectivity
    pdfs = np.asarray([dist(idx, method="inverse", zero_val=0.0, inverse_pow=conn_pow) for idx in indices])
    pdfs /= np.sum(pdfs)
    W = circular_connectivity(N, p, spatial_distribution=rv_discrete(values=(indices, pdfs)), homogeneous_weights=False)

    # initialize model
    net = Network(dt=dt, device="cuda:0")
    net.add_diffeq_node("rs", node=f"config/ik_snn/rs", weights=W, source_var="s", target_var="s_in",
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

    # calculate the network dimensionality
    dims = []
    for i, s in enumerate(signals):
        corr_net = np.cov(s)
        eigs = np.abs(np.linalg.eigvals(corr_net))
        dims.append(np.sum(eigs) ** 2 / np.sum(eigs ** 2))

    # calculate the network sequentiality
    seqs = []
    for i, s in enumerate(signals):
        seqs.append(sequentiality(s, neighborhood=seq_range, max_lag=margin))

    # calculate the network covariance matrices
    cs = []
    for i, s in enumerate(signals):
        cs.append(get_c(s, alpha=1e-4))

    # calculate the network kernel
    s_mean = np.mean(signals, axis=0)
    s_var = np.mean([s_i - s_mean for s_i in signals], axis=0)
    C_inv = np.linalg.inv(np.mean(cs, axis=0))
    w = C_inv @ s_mean
    K = s_mean.T @ w
    G = s_var.T @ w

    # calculate the network response on test data
    test_onsets = np.random.randint(low=np.min(stim_onsets), high=np.max(stim_onsets), size=n_tests)
    test_signals, _ = get_signals(list(test_onsets), cycle_steps, sr, net, y0, inp_indices, sigma=sigma)

    # store results
    results["dimensionality"].append(np.mean(dims))
    results["sequentiality"].append(np.mean(seqs))
    results["Ks"].append(K)
    results["Gs"].append(G)
    results["test_signals"].append(test_signals)
    results["test_onsets"].append(np.round(dt*2.0*np.pi*test_onsets/T, decimals=2))

    print(f"Finished {(i+1)} of {len(alphas)} jobs.")

    # plot results
    # _, axes = plt.subplots(nrows=2, figsize=(12, 5))
    # s_all = np.concatenate(signals, axis=1)
    # inp_all = np.concatenate(inputs, axis=0)
    # s_all /= np.max(s_all)
    # inp_all /= np.max(inp_all)
    # ax = axes[0]
    # ax.plot(np.mean(s_all, axis=0), label="s")
    # ax.plot(inp_all, label="I_ext")
    # ax.legend()
    # ax.set_xlabel("time")
    # ax.set_title("Mean signal")
    # ax = axes[1]
    # im = ax.imshow(s_all, aspect="auto", interpolation="none")
    # plt.colorbar(im, ax=ax, shrink=0.4)
    # ax.set_xlabel('time')
    # ax.set_ylabel('neurons')
    # ax.set_title(f"Seq = {np.mean(seqs)}, Dim = {np.mean(dims)}")
    # plt.tight_layout()
    # _, axes = plt.subplots(ncols=2, figsize=(12, 6))
    # ax = axes[0]
    # ax.imshow(K, aspect="auto", interpolation="none")
    # ax.set_xlabel("lags")
    # ax.set_ylabel("lags")
    # ax.set_title(f"K")
    # ax = axes[1]
    # ax.imshow(G, aspect="auto", interpolation="none")
    # ax.set_xlabel("lags")
    # ax.set_ylabel("lags")
    # ax.set_title(f"G")
    # plt.tight_layout()
    # plt.show()

# save results
fname = f"snn_entrainment"
with open(f"{tdir}/{fname}_{cond}.pkl", "wb") as f:
    pickle.dump(results, f)
    f.close()
