import numpy as np
import pickle
from rectipy.utility import input_connections, random_connectivity, circular_connectivity
from scipy.stats import rv_discrete, cauchy
from scipy.ndimage import convolve1d
import sys


def dist(x: int, method: str = "inverse") -> float:
    if method == "inverse":
        return 1/x if x > 0 else 1
    if method == "inverse_squared":
        return 1/x**2 if x > 0 else 1
    if method == "exp":
        return np.exp(-x)
    else:
        raise ValueError("Invalid method.")


def lorentzian(n: int, eta: float, delta: float, lb: float, ub: float):
    samples = np.zeros((n,))
    for i in range(n):
        s = cauchy.rvs(loc=eta, scale=delta)
        while s <= lb or s >= ub:
            s = cauchy.rvs(loc=eta, scale=delta)
        samples[i] = s
    return samples


def alpha_kernel(t: np.ndarray, tau: float):
    """Alpha kernel function.

    :param t: time in arbitrary units. Can be a vector or a single scalar.
    :param tau: Time constant of the alpha kernel. Must be a scalar.
    :return: Value of the kernel function at each entry in `t`.
    """
    return g * t * np.exp(1 - t / tau) / tau


# file name
###########

cond = sys.argv[1]
fname = f'fs_dc_{cond}'

# simulation parameters
#######################

# simulation time
T = 101000.0
dt = 1e-2
sr = 100
cutoff = 1000.0

# extrinsic input default
alpha = 10.0
stim_rate = 0.08
def_rate = 0.004
isi_std = 50.0

# Default model parameters
##########################

# network default
N = 1000
p = 0.1
m = 3

# node default
C = 20.0
k = 1.0
v_r = -55.0
v_t = -40.0
Delta = 1.0
eta = 2.0
a = 0.2
b = 0.025
d = 0.0
g = 10.0
E_r = -65.0
tau_s = 10.0
v_spike = 1000.0
v_reset = -1000.0

# overwrite any default parameters with sweep condition
#######################################################

sweep = {}
for idx in range(2, len(sys.argv), 2):
    exec(f"{sys.argv[idx]} = {sys.argv[idx+1]}")
    sweep[sys.argv[idx]] = sys.argv[idx+1]

# set up final network variables
################################

# setup connectivity matrix
indices = np.arange(0, N, dtype=np.int32)
pdfs = np.asarray([dist(idx, method="inverse") for idx in indices])
pdfs /= np.sum(pdfs)
if "connectivity" in sweep and sweep["connectivity"] == "circcular":
    W = circular_connectivity(N, p, spatial_distribution=rv_discrete(values=(indices, pdfs)))
else:
    W = random_connectivity(N, N, p, normalize=True)

# setup input matrix
p_in = 0.2
W_in = input_connections(N, m, p_in)
print(np.sum(W_in, axis=0))

# collect remaining model parameters
node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": lorentzian(N, v_t, Delta, v_r, 2 * v_t - v_r),
             "eta": eta, "tau_u": 1/a, "b": b, "kappa": d, "g": g, "E_r": E_r, "tau_s": tau_s, "v": v_t}

# define stimulation times
steps = int(T/dt)
mean_isi = int(1000.0/dt)
start = cutoff
stim_dur = int(100.0/dt)
stim_times = []
idx = start
while idx < steps:
    if idx > 0:
        stim_times.append(idx)
    idx += mean_isi + int(np.random.randn() * isi_std)

# create background noise
inp = np.zeros((steps, m))
for c in range(m):
    inp[:, c] = np.random.poisson(def_rate, (steps,))

# create stimuli
stim_channels = []
for idx in stim_times:

    # add poisson spike trains to input array at defined stimulation times
    if idx+stim_dur < steps:
        idx_time = np.arange(idx, idx+stim_dur, dtype=np.int32)
        n_inputs = 1  #np.random.randint(1, m+1)
        channels = np.random.randint(0, m, size=n_inputs)
        for c in channels:
            spike_train = np.random.poisson(stim_rate, (stim_dur,))
            inp[idx_time, c] = spike_train
        stim_channels.append(channels)

# convolve input spikes with synaptic alpha kernel
kernel = alpha_kernel(np.arange(0, 200.0/dt), tau=tau_s/dt)
kernel /= np.max(kernel)
for c in range(m):
    inp[:, c] = convolve1d(inp[:, c], weights=kernel)

# store data
data = {}
data['T'] = T
data['dt'] = dt
data['sr'] = sr
data['cutoff'] = cutoff
data['N'] = N
data['p'] = p
data['W'] = W
data['W_in'] = W_in
data['inp'] = inp * alpha
data['stim_times'] = stim_times
data['stim_channels'] = stim_channels
data['stim_dur'] = int(stim_dur/sr)
data['node_vars'] = node_vars
data['sweep'] = sweep
data['additional_params'] = {"v_reset": v_reset, "v_spike": v_spike}
pickle.dump(data, open(f"config/{fname}_config.pkl", 'wb'))

# plotting
##########

import matplotlib.pyplot as plt

# input data
plt.imshow(inp.T, aspect='auto', interpolation='none')
plt.colorbar()
plt.show()

# weight matrix
plt.imshow(W, aspect='auto', interpolation='none')
plt.colorbar()
plt.show()

# input weights
plt.imshow(W_in, aspect='auto', interpolation='none')
plt.colorbar()
plt.show()
