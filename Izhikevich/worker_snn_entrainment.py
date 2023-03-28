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
from scipy.signal import correlate
from time import perf_counter


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


def get_kernel(X: np.ndarray, alpha: float = 1e-12):
    """
    """
    X_t = X.T
    return X @ np.linalg.inv(X_t @ X + alpha*np.eye(X.shape[1])) @ X_t


# define parameters
###################

# model parameters
N = 500
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
cond = 55
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
T = 5000.0
dt = 1e-2
sr = 10
p_in = 0.1
omega = 0.0035
steps = int(T/dt)
n_inputs = int(p_in*N)
center = int(N*0.5)
inp_indices = np.arange(center-int(0.5*n_inputs), center+int(0.5*n_inputs))
inp_dist = np.zeros((N,))
inp_dist[inp_indices] = 1.0
cutoff = int(2000.0/(dt*sr))

# define stimulation signal
time = np.linspace(0, T, steps)
driver_tmp = np.sin(2.0*np.pi*omega*time)
driver = np.zeros_like(driver_tmp)
driver[driver_tmp > 0.9] = 1.0

# downsample driver and calculate maximum correlation between driver and itself
driver_ds = driver[::sr]
max_corr = np.max(correlate(driver_ds, driver_ds, mode="valid"))

# infer stimulation onsets
margin = 10
ds = 2
driver_diff = np.diff(driver_ds[cutoff::ds])
stim_onsets = np.argwhere(driver_diff > 0.1).squeeze()
min_isi = np.min(np.abs(np.diff(stim_onsets))) - margin

# other analysis parameters
sigma = 50
max_lag = int(0.5/(omega*dt*sr*ds))
kernel_lag = 5

# define lorentzian of etas
thetas = lorentzian(N, eta=v_t, delta=Delta, lb=v_r, ub=0.0)

# define connectivity
indices = np.arange(0, N, dtype=np.int32)
pdfs = np.asarray([dist(idx, method="inverse", zero_val=0.0, inverse_pow=1.5) for idx in indices])
pdfs /= np.sum(pdfs)
W = circular_connectivity(N, p, spatial_distribution=rv_discrete(values=(indices, pdfs)), homogeneous_weights=False)

# simulation
############

# adjust parameters according to sweep condition
for param, v in zip([p1, p2], [v1, v2]):
    exec(f"{param} = {v}")

# prepare results storage
alphas = np.linspace(1e-1, 2e-1, num=20)
results = {"sweep": {p1: v1, p2: v2}, "T": T, "dt": dt, "sr": sr, "p": p, "correlations": [], "alphas": alphas, "W": W,
           "thetas": thetas, "input_indices": inp_indices, "dimensionality": [], "sequentiality": []}
for i, alpha in enumerate(alphas):

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
    t0 = perf_counter()
    obs = net.run(inputs=inp, sampling_steps=sr, record_output=True, verbose=False)
    res = obs["out"].iloc[cutoff:, :].values
    s = gaussian_filter1d(res, sigma=50, axis=0).T
    t1 = perf_counter()
    print(f"Simulation time: {t1-t0} s.")

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(s, aspect=4.0, interpolation="none")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xlabel('time')
    ax.set_ylabel('neurons')
    plt.show()

    # calculate the correlation between the input and each network unit
    t0 = perf_counter()
    corr_driver = np.asarray([corr(driver_ds, s[idx, :], method="auto")/max_corr for idx in range(N)])
    t1 = perf_counter()
    print(f"Corr(driver, net) time: {t1 - t0} s.")

    # calculate the network dimensionality
    t0 = perf_counter()
    corr_net = np.cov(s)
    eigs = np.abs(np.linalg.eigvals(corr_net))
    dim = np.sum(eigs)**2/np.sum(eigs**2)
    t1 = perf_counter()
    print(f"Dimensionality time: {t1 - t0} s.")

    # calculate the network kernel
    t0 = perf_counter()
    kernel_performance = []
    s_tmp = s[:, ::ds]
    for sidx in stim_onsets:
        if s_tmp.shape[1] - sidx > min_isi:
            K = get_kernel(s_tmp[:, sidx:sidx+min_isi])
            diff = (K - np.eye(K.shape[0]))**2
            decoding_performance = []
            lag = 0
            while lag < max_lag:
                decoding_performance.append(1.0/np.sqrt(np.mean(diff[lag:lag+kernel_lag, :])))
                lag += kernel_lag
            kernel_performance.append(decoding_performance)
    t1 = perf_counter()
    print(f"Kernel analysis time: {t1 - t0} s.")

    # store results
    results["correlations"].append(corr_driver)
    results["dimensionality"].append(dim)
    results["sequentiality"].append(np.mean(kernel_performance, axis=0))

    # plot results
    fig, axes = plt.subplots(nrows=2, figsize=(10, 4))
    ax = axes[0]
    ax.set_title(f"Dim = {dim}")
    ax.plot(corr_driver, label="correlation with driver")
    ax.plot(inp_dist, label="input")
    ax.set_xlabel("neurons")
    ax.set_ylabel("correlation")
    ax.legend()
    ax = axes[1]
    ax.plot(results["sequentiality"][-1])
    ax.set_xlabel("lags (ms)")
    ax.set_ylabel("performance")
    plt.tight_layout()
    plt.show()

# save results
fname = f"snn_entrainment"
with open(f"{tdir}/{fname}_{cond}.pkl", "wb") as f:
    pickle.dump(results, f)
    f.close()
