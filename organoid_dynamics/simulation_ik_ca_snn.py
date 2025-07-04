import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['backend'] = 'TkAgg'
from rectipy import Network, random_connectivity
import sys
sys.path.append("~/PycharmProjects/DynamicalSystems/reservoir_computing")
from custom_functions import *
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# define parameters
###################

# model parameters
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
Delta = 10.0
eta = 0.0
b = -10.0
kappa = 0.001
alpha = 0.1
gamma = 200.0
theta = 0.02
mu = 1.0
tau_a = 2000.0
tau_u = 100.0
g = 30.0
E_r = 0.0
I_ext = 60.0

noise_lvl = 50.0
noise_sigma = 160.0

N = 200
v_spike = 1000.0
v_reset = -1000.0

# define inputs
T = 10000.0
cutoff = 0.0
dt = 1e-2
dts = 1e-1
inp = np.zeros((int(T/dt), N)) + I_ext
# for i in range(N):
#     noise = noise_lvl*np.random.randn(inp.shape[0])
#     noise = gaussian_filter1d(noise, sigma=noise_sigma)
#     inp[:, i] += noise

# define lorentzian of etas
etas = lorentzian(N, loc=eta, scale=Delta, lb=-1e3, ub=1e3)

params = {
    'C': C, 'k': k, 'v_r': v_r, 'v_t': v_t, 'eta': etas, 'kappa': kappa, 'alpha': alpha,
    'tau_a': tau_a, 'tau_u': tau_u, 'g': g, 'E_r': E_r, 'b': b, 's': gamma, 'theta': theta, 'mu': mu
}

# initialize model
op = "ik_ca_op"
net = Network(dt=dt, device="cpu")
net.add_diffeq_node("snn", f"config/ik_snn/ik_ca", N=N,
                    input_var="I_ext", output_var="s", spike_var="spike", reset_var="v", to_file=True,
                    node_vars={f"all/{op}/{var}": val for var, val in params.items()}, op=op,
                    spike_reset=v_reset, spike_threshold=v_spike, verbose=False, clear=False, device="cpu")

# perform simulation
obs = net.run(inputs=inp, sampling_steps=int(dts/dt), record_output=True, verbose=True, cutoff=int(cutoff/dt))
res_snn = obs.to_dataframe("out")

# detect spikes
spikes = []
for i in range(N):
    signal = res_snn.iloc[:, i].values
    s = np.zeros_like(signal)
    peaks, _ = find_peaks(signal, height=20.0)
    s[peaks] = 1.0/dts
    spikes.append(s)
fr = np.mean(spikes, axis=0) * 1e3
fr = gaussian_filter1d(fr, sigma=50)

fig, axes = plt.subplots(nrows=2, figsize=(12, 6))
ax = axes[0]
im = ax.imshow(np.asarray(spikes), aspect="auto", interpolation="none", cmap="Greys")
plt.colorbar(im, ax=ax)
ax.set_xlabel("steps")
ax.set_ylabel("neurons")
ax = axes[1]
ax.plot(fr)
ax.set_xlabel("steps")
ax.set_ylabel("r (Hz)")
plt.tight_layout()
plt.show()
