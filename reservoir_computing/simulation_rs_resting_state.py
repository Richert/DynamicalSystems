from rectipy import Network, random_connectivity, circular_connectivity
import numpy as np
import pickle
from scipy.stats import rv_discrete, cauchy
import sys


def lorentzian(n: int, eta: float, delta: float, lb: float, ub: float):
    samples = np.zeros((n,))
    for i in range(n):
        s = cauchy.rvs(loc=eta, scale=delta)
        while s <= lb or s >= ub:
            s = cauchy.rvs(loc=eta, scale=delta)
        samples[i] = s
    return samples


def correlation(s: np.ndarray):
    s = s - np.mean(s)
    s = s / np.std(s)
    c = np.corrcoef(s, rowvar=False)
    c[np.isnan(c)] = 0.0
    c[np.eye(c.shape[0]) > 0.0] = 0.0
    return c


def get_dim(s: np.ndarray):
    s = s - np.mean(s)
    s = s / np.std(s)
    cov = s.T @ s
    cov[np.eye(cov.shape[0]) > 0] = 0.0
    eigs = np.abs(np.linalg.eigvals(cov))
    return np.sum(eigs)**2/np.sum(eigs**2)


def dist(x: int, method: str = "inverse") -> float:
    if method == "inverse":
        return 1/x if x > 0 else 1
    if method == "inverse_squared":
        return 1/x**2 if x > 0 else 1
    if method == "exp":
        return np.exp(-x)
    else:
        raise ValueError("Invalid method.")


# model definition
##################

# default network parameters
N = 1000
p = 0.01
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

# simulation settings
T = 11000.0
dt = 1e-2
steps = int(T/dt)
sampling_steps = 100
I_ext = np.zeros((steps, 1))

# handle user-supplied arguments
cond = sys.argv[1]
params = {}
for idx in range(2, len(sys.argv), 2):
    exec(f"{sys.argv[idx]} = {sys.argv[idx+1]}")
    params[sys.argv[idx]] = sys.argv[idx+1]

# simulation
############

# create connectivity matrix
connectivity = "exponential"
indices = np.arange(0, N, dtype=np.int32)
pdfs = np.asarray([dist(idx, method="inverse_squared") for idx in indices])
pdfs /= np.sum(pdfs)
if connectivity == "circular":
    W = circular_connectivity(N, p, spatial_distribution=rv_discrete(values=(indices, pdfs)))
else:
    W = random_connectivity(N, N, p, normalize=True)

# create background current distribution
thetas = lorentzian(N, v_t, Delta, v_r, 2 * v_t - v_r)

# collect remaining model parameters
node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": thetas, "eta": eta, "tau_u": 1 / a, "b": b, "kappa": d, "g": g,
             "E_r": E_r, "tau_s": tau_s, "v": v_t}

# initialize model
net = Network.from_yaml("config/ik/rs", weights=W, source_var="s",
                        target_var="s_in", input_var="I_ext", output_var="s", spike_var="spike", spike_def="v",
                        to_file=False, node_vars=node_vars.copy(), op="rs_op", spike_reset=v_reset,
                        spike_threshold=v_spike, dt=dt, verbose=False, clear=True, device="cuda:0")

# simulation
obs = net.run(inputs=I_ext, sampling_steps=sampling_steps, record_output=True, verbose=False)

# results storage
results = {"s": obs["out"], "J": W, "thetas": thetas, "T": T, "dt": dt, "sr": sampling_steps, "condition": params}
pickle.dump(results, open(f"results/rs_resting_state_{cond}.pkl", "wb"))

# plotting
##########

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(12, 8))
grid = GridSpec(nrows=3, ncols=2, figure=fig)

cutoff = 1000
signal = results["s"].values[cutoff:, :]

# spike raster plot
ax = fig.add_subplot(grid[0, :])
im = ax.imshow(signal.T, aspect=2.5, vmin=0.0, vmax=np.max(signal), interpolation="none")
ax.set_xlabel("time")
ax.set_ylabel("neurons")
ax.set_title(", ".join([f'{key} = {val}' for key, val in params.items()]))
plt.colorbar(im, ax=ax, shrink=0.5)
plt.title(f"Spiking activity (eta = {eta}, g = {g})")

# average activity
ax = fig.add_subplot(grid[1, :])
ax.plot(np.mean(signal, axis=-1))
ax.set_xlabel("time (ms)")
ax.set_ylabel("s")
ax.set_title("mean-field dynamics")

# correlation
ax = fig.add_subplot(grid[2, 0])
c = correlation(signal)
im = ax.imshow(c, aspect=1.0, vmin=0.0, vmax=np.max(c), interpolation="none")
ax.set_xlabel("neurons")
ax.set_ylabel("neurons")
plt.colorbar(im, ax=ax, shrink=0.5)
plt.title(f"Correlation (dim = {get_dim(signal)})")

# W
ax = fig.add_subplot(grid[2, 1])
im = ax.imshow(W, aspect=1.0, vmin=0.0, vmax=np.max(W), interpolation="none")
ax.set_xlabel("neurons")
ax.set_ylabel("neurons")
plt.colorbar(im, ax=ax, shrink=0.5)
plt.title(f"Synaptic coupling")

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/rs_resting_state_{cond}.pdf')
plt.show()
