import matplotlib.pyplot as plt
import numpy as np
from pyrates import CircuitTemplate, clear
from numba import njit

# choose neuron type
neuron_type = 'spn'

# model parameters
C = 80.0
k = 0.13
v_r = -81.0
v_t = -45.0
eta = 100.0
Delta = 0.9
kappa = 150.0
a = 0.01
b = 5.0
tau_r = 2.0
tau_d = 8.0
g_i = 4.0
E_i = -60.0
node_vars = {"C": C, "k": k, "v_r": v_r, "v_t": v_t, "tau_u": 1/a, "b": b, "kappa": kappa, "eta": eta,
             "g_i": g_i, "E_i": E_i, "tau_r": tau_r, "tau_d": tau_d, "v": v_t, "Delta": Delta}
edge_vars = [(f'p/{neuron_type}_op/s', f'p/{neuron_type}_op/s_i', {'weight': 1.0})]

# load model template
template = CircuitTemplate.from_yaml(f'config/mf/{neuron_type}')

# update template parameters
template.update_var(node_vars={f"p/{neuron_type}_op/{key}": val for key, val in node_vars.items()}, edge_vars=edge_vars)

# set pyrates-specific parameters
cutoff = 100.0
T = 1000.0 + cutoff
start = 300.0 + cutoff
stop = 600.0 + cutoff
dt = 1e-3
dts = 1e-1
inp = np.zeros((int(T/dt),))
inp[int(start/dt):int(stop/dt)] -= 0.3
backend = 'default'
solver = 'euler'
out_vars = ['r', 's', 'v']
kwargs = {'vectorize': False, 'float_precision': 'float64', 'decorator': njit, 'fastmath': False}

# perform simulation
results = template.run(simulation_time=T, step_size=dt, backend=backend, solver=solver, sampling_step_size=dts,
                       inputs={f'p/{neuron_type}_op/d': inp},
                       outputs={v: f'p/{neuron_type}_op/{v}' for v in out_vars}, cutoff=10.0)
clear(template)

# plot simulated signal
fig, axes = plt.subplots(figsize=(10, 2*len(out_vars)), nrows=len(out_vars))
for i, v in enumerate(out_vars):
    ax = axes[i]
    ax.plot(results[v])
    ax.set_xlabel('time (ms)')
    ax.set_ylabel(v)
plt.show()
