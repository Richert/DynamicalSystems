import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['backend'] = 'TkAgg'
from rectipy import Network, random_connectivity
import sys
sys.path.append("~/PycharmProjects/DynamicalSystems/reservoir_computing")
from custom_functions import *
from scipy.ndimage import gaussian_filter1d

# define parameters
###################

# model parameters
N = 1000
p = 0.5
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
Delta = 0.1
eta = 0.0
u0 = 0.5
kappa = 2.0
tau_u = 50.0
tau_w = 500.0
g = 20.0
E_r = 0.0
tau_s = 6.0
s_ext = 67.0
noise_lvl = 59.0
noise_sigma = 50.0
v_spike = 1000.0
v_reset = -1000.0

# define inputs
T = 10000.0
cutoff = 0.0
dt = 1e-2
dts = 1e-1
inp = np.zeros((int(T/dt),)) + s_ext
noise = noise_lvl*np.random.randn(inp.shape[0])
noise = gaussian_filter1d(noise, sigma=noise_sigma)
inp += noise

# define lorentzian of etas
thetas = lorentzian(N, loc=v_t, scale=Delta, lb=v_r, ub=2*v_r-v_t)

# define connectivity
spatial_dim = 1
if spatial_dim == 0:
    W = random_connectivity(N, N, p, normalize=True)
elif spatial_dim == 1:
    W = circular_connectivity(N, N, p, homogeneous_weights=True, scale=1.0)
else:
    W = spherical_connectivity(N, N, p, homogeneous_weights=True, scale=1.0)

params = {
    'C': C, 'k': k, 'v_r': v_r, 'v_t': v_t, 'Delta': Delta, 'eta': eta, 'u0': u0, 'kappa': kappa, 'tau_u': tau_u,
    'tau_w': tau_w, 'g': g, 'E_r': E_r, 'tau_s': tau_s
}

# initialize model
op = "ik_stp_op"
net = Network(dt=dt, device="cpu")
net.add_diffeq_node("snn", f"config/ik_snn/{op}", weights=W, source_var="s", target_var="s_in",
                    input_var="I_ext", output_var="s", spike_var="spike", reset_var="v", to_file=False,
                    node_vars={f"p/{op}/{var}": val for var, val in params.items()}, op=op,
                    spike_reset=v_reset, spike_threshold=v_spike, verbose=False, clear=True, device="cpu")

# perform simulation
obs = net.run(inputs=inp, sampling_steps=int(dts/dt), record_output=True, verbose=True, cutoff=int(cutoff/dt))
res_snn = obs.to_dataframe("out")

