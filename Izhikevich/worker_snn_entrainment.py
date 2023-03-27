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
from scipy.signal import correlate, correlation_lags


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


def cross_corr(N: int, signals: np.ndarray, method: str = 'direct', max_lag: int = 100) -> np.ndarray:
    C = np.zeros((N, N))
    padding = list(np.zeros((max_lag,)))
    for n2 in range(N):
        s2 = np.asarray(padding + list(signals[n2]) + padding)
        for n1 in range(n2+1, N):
            s1 = signals[n1]
            C[n1, n2] = np.max(correlate(s1, s2, mode="valid", method=method))
            C[n2, n1] = C[n1, n2]
    return C


def sequentiality_calculation(N: int, signals: np.ndarray, lags: np.ndarray, zero_lag: int, method: str = 'direct') -> tuple:
    sym = 0
    asym = 0
    max_lag = np.max(lags)
    padding = list(np.zeros((max_lag,)))
    for n1 in range(N):
        s1 = np.asarray(padding + list(signals[n1]) + padding)
        for n2 in range(N):
            cc = correlate(s1, signals[n2], mode="valid", method=method)
            for l in lags:
                sym += (cc[zero_lag + l] - cc[zero_lag - l]) ** 2
                asym += (cc[zero_lag + l] + cc[zero_lag - l]) ** 2
    return sym, asym


def sequentiality(signals: np.ndarray, **kwargs) -> float:
    """Estimates the sequentiality of the dynamics of a system using the method proposed by Bernacchia et al. (2022).

    :param signals: `N x T` matrix containing the dynamics of `N` units sampled at `T` time steps.
    :param kwargs: Additional keyword arguments to be passed to the decorator function.
    :return: Estimate of the sequentiality of the system dynamcis.
    """

    # preparations
    N = signals.shape[0]
    m = signals.shape[1]
    lags = kwargs.pop("lags", correlation_lags(m, m, mode='valid'))
    lags_pos = lags[lags > 0]
    zero_lag = np.argwhere(lags == 0)[0]

    # sum up cross-correlations over neurons and lags
    sym, asym = sequentiality_calculation(N, signals, lags_pos, zero_lag)

    # calculate sequentiality
    return np.sqrt(sym/asym)


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
# wdir = "config"
# tdir = "results"

# sweep condition
# cond = 350
p1 = "Delta"
p2 = "trial"

# parameter sweep definition
with open(f"{wdir}/bump_sweep.pkl", "rb") as f:
    sweep = pickle.load(f)
    v1s = sweep[p1]
    v2s = sweep[p2]
    f.close()
vals = [(v1, v2) for v1 in v1s for v2 in v2s]
v1, v2 = vals[int(cond)]
print(f"Condition: {p1} = {v1},  {p2} = {v2}")

# define inputs
T = 5000.0
dt = 1e-2
sr = 10
p_in = 0.1
omega = 0.0035
steps = int(T/dt)
inp = np.zeros((steps, N))
time = np.linspace(0, T, steps)
driver = np.sin(2.0*np.pi*omega*time)
n_inputs = int(p_in*N)
center = int(N*0.5)
inp_indices = np.arange(center-int(0.5*n_inputs), center+int(0.5*n_inputs))
inp[driver > 0.3, inp_indices] = 5e-3
cutoff = int(2000.0/(dt*sr))

# cross-correlation parameters
sigma = 50
max_lag = int(0.5/(omega*dt*sr))
lags = np.arange(0, max_lag)

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
alphas = np.linspace(1e-3, 2e-2, num=20)
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
                            dt=dt, verbose=False, clear=True, device="cpu")

    # perform simulation
    obs = net.run(inputs=inp, sampling_steps=sr, record_output=True, verbose=False)
    res = obs["out"].iloc[cutoff:, :]
    s = gaussian_filter1d(res, sigma=50, axis=0)

    # calculate the correlation between the input and each network unit
    corr_driver = np.asarray([corr(driver[::sr], s[:, idx]) for idx in range(N)])

    # calculate the network dimensionality
    corr_net = cross_corr(N, s, max_lag=max_lag)
    eigs = np.linalg.eigvals(corr_net)
    dim = np.sum(eigs)**2/np.sum(eigs**2)

    # calculate the network sequentiality
    seq = sequentiality(s, lags=lags)

    # store results
    results["correlations"].append(corr_driver)
    results["dimensionality"].append(dim)
    results["sequentiality"].append(seq)

    # plot results
    # fig, axes = plt.subplots(nrows=2, figsize=(12, 8))
    # ax = axes[0]
    # im = ax.imshow(s.T, aspect=4.0, interpolation="none")
    # plt.colorbar(im, ax=ax, shrink=0.8)
    # ax.set_xlabel('time')
    # ax.set_ylabel('neurons')
    # ax = axes[1]
    # ax.plot(target_dist, label="target")
    # ax.plot(population_dist, label="SNN")
    # ax.set_xlabel("neurons")
    # ax.set_ylabel("probability")
    # plt.tight_layout()
    # plt.show()

# save results
fname = f"snn_entrainment"
with open(f"{tdir}/{fname}_{cond}.pkl", "wb") as f:
    pickle.dump(results, f)
    f.close()
