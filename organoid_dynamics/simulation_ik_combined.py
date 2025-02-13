from pyrates import CircuitTemplate, clear_frontend_caches
import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.rcParams['backend'] = 'TkAgg'
import numba as nb
from rectipy import Network
import sys
sys.path.append("~/PycharmProjects/DynamicalSystems/reservoir_computing")
from custom_functions import *
from scipy.ndimage import gaussian_filter
from scipy.stats import poisson

# define parameters
###################

model = "ik_sfa"
op = f"{model}_op"

# model parameters
N = 200
p = 0.2
spatial_dim = 0
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
Delta = 0.1
eta = 0.0
kappa = 8.0
tau_u = 500.0
g = 20.0
E_r = 0.0
tau_s = 6.0
s_ext = 0.0
noise_lvl = 69.0
noise_sigma = 2.0
neuron_sigma = 4.0
v_spike = 1000.0
v_reset = -1000.0

# define inputs
T = 2500.0
cutoff = 500.0
dt = 1e-2
dts = 1e-1
inp = np.zeros((int(T/dt), N)) + s_ext
noise = poisson.rvs(mu=noise_lvl, size=(inp.shape[0], N))
inp += gaussian_filter(noise, sigma=(noise_sigma, neuron_sigma))

# _, ax = plt.subplots(figsize=(12, 4))
# im = ax.imshow(inp.T, aspect="auto", interpolation="none")
# plt.colorbar(im, ax=ax)
# plt.show()

params = {
    'C': C, 'k': k, 'v_r': v_r, 'v_t': v_t, 'Delta': Delta, 'eta': eta, 'kappa': kappa, 'tau_u': tau_u,
    'g': g, 'E_r': E_r, 'tau_s': tau_s
}

# run the mean-field model
##########################

# initialize model
ik = CircuitTemplate.from_yaml(f"config/ik_mf/{model}")

# update parameters
ik.update_var(node_vars={f"p/{op}/{var}": val for var, val in params.items()})

# run simulation
res_mf = ik.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='euler',
                outputs={'s': f'p/{op}/s'}, inputs={f'p/{op}/I_ext': np.mean(inp, axis=1)},
                decorator=nb.njit, fastmath=True, clear=True)

# run the SNN model
###################

clear_frontend_caches(clear_template_cache=True, clear_ir_cache=True)
params.pop("Delta")
params["v_t"] = lorentzian(N, loc=v_t, scale=Delta, lb=v_r, ub=2*v_t-v_r)

# define connectivity
if spatial_dim == 0:
    W = random_connectivity(N, N, p, normalize=True)
elif spatial_dim == 1:
    W = circular_connectivity(N, N, p, homogeneous_weights=True, scale=0.5, normalize=True)
else:
    W = spherical_connectivity(N, N, p, homogeneous_weights=True, scale=0.5, normalize=True)

_, ax = plt.subplots(figsize=(6, 6))
ax.imshow(W, interpolation="none", aspect="auto")
plt.show()

# initialize model
net = Network(dt=dt, device="cpu")
net.add_diffeq_node("snn", f"config/ik_snn/{model}", weights=W, source_var="s", target_var="s_in",
                    input_var="I_ext", output_var="s", spike_var="spike", reset_var="v", to_file=False,
                    node_vars=params, op=op, spike_reset=v_reset, spike_threshold=v_spike, verbose=False,
                    clear=True, device="cpu")

# perform simulation
obs = net.run(inputs=inp, sampling_steps=int(dts/dt), record_output=True, verbose=True, cutoff=int(cutoff/dt))
res_snn = obs.to_dataframe("out")

# plot results
fig, axes = plt.subplots(nrows=2, figsize=(12, 6))
ax = axes[0]
ax.plot(res_mf.index, res_mf["s"], label="FRE")
ax.plot(res_mf.index, np.mean(res_snn, axis=1), label="SNN")
ax.set_ylabel(r'$s(t)$')
ax.set_xlabel("time (ms)")
ax.set_title("Macroscopic Dynamics")
ax.legend()
ax = axes[1]
ax.imshow(res_snn.T, aspect="auto", interpolation="none", cmap="Greys")
ax.set_xlabel("time")
ax.set_ylabel("neurons")
ax.set_title("Microscopic Dynamics")
plt.tight_layout()
plt.show()

# save results
# pickle.dump({'mf': res_mf, "snn": res_snn}, open("results/rs_high_sfa.p", "wb"))
