from rectipy import Network, random_connectivity, input_connections
import numpy as np
import pickle
from scipy.stats import cauchy
from scipy.ndimage import gaussian_filter1d


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


# model definition
##################

# network parameters
N = 1000
p = 0.1
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
Delta = 1.0
eta = 40.0
a = 0.03
b = -2.0
d = 200.0
g = 8.0
E_r = 0.0
tau_r = 2.0
tau_d = 8.0
v_spike = 1000.0
v_reset = -1000.0

# simulation settings
T = 11000.0
dt = 1e-1
steps = int(T/dt)
sampling_steps = 10

# input definition
p_in = 0.2
alpha = 200.0
sigma = 40
stimuli = np.random.randn(steps, 1)

# generate input
I_ext = np.zeros_like(stimuli)
I_ext[:, 0] = gaussian_filter1d(input=stimuli[:, 0], sigma=sigma)
I_ext[:, 0] /= np.max(np.abs(I_ext[:, 0]))

# simulation
############

# create connectivity matrices
J = random_connectivity(N, N, p, normalize=True)
W_in = input_connections(N, stimuli.shape[1], p_in, variance=1.0, zero_mean=True)

# create background current distribution
thetas = lorentzian(N, v_t, Delta, v_r, 2 * v_t - v_r)

# collect remaining model parameters
node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": thetas, "eta": eta, "tau_u": 1 / a, "b": b, "kappa": d, "g": g,
             "E_r": E_r, "tau_r": tau_r, "tau_d": tau_d, "v": v_t}

# initialize model
net = Network.from_yaml("neuron_model_templates.spiking_neurons.ik.ik_biexp", weights=J, source_var="s",
                        target_var="s_in", input_var="I_ext", output_var="s", spike_var="spike", spike_def="v",
                        to_file=False, node_vars=node_vars.copy(), op="ik_biexp_op", spike_reset=v_reset,
                        spike_threshold=v_spike, dt=dt, verbose=False, clear=True, device="cuda:0")
net.add_input_layer(stimuli.shape[1], W_in, trainable=False)

# simulation
obs = net.run(inputs=I_ext * alpha, sampling_steps=sampling_steps, record_output=True, verbose=False)

# results storage
results = {"s": obs["out"], "J": J, "thetas": thetas, "T": T, "dt": dt, "sr": sampling_steps, "eta": eta, "g": g,
           "W_in": W_in}
pickle.dump(results, open("results/rs_noise_driven.pkl", "wb"))

# plotting
##########

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(12, 8))
grid = GridSpec(nrows=2, ncols=2, figure=fig)

cutoff = 1000
signal = results["s"].values[cutoff:, :]

# spike raster plot
ax = fig.add_subplot(grid[0, :])
im = ax.imshow(signal.T, aspect=2.5, vmin=0.0, vmax=np.max(signal), interpolation="none")
ax.set_xlabel("time")
ax.set_ylabel("neurons")
plt.colorbar(im, ax=ax, shrink=0.5)
plt.title(f"Spiking activity (eta = {eta}, g = {g})")

# correlation
ax = fig.add_subplot(grid[1, 0])
c = correlation(signal)
im = ax.imshow(c, aspect=1.0, vmin=0.0, vmax=np.max(c), interpolation="none")
ax.set_xlabel("neurons")
ax.set_ylabel("neurons")
plt.colorbar(im, ax=ax, shrink=0.5)
plt.title(f"Correlation (dim = {get_dim(signal)})")

# average activity
ax = fig.add_subplot(grid[1, 1])
ax.plot(np.mean(signal, axis=-1))
ax.set_xlabel("time (ms)")
ax.set_ylabel("s")
ax.set_title("mean-field dynamics")

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/nd_eta_{int(eta)}_g_{int(g)}.pdf')
plt.show()
