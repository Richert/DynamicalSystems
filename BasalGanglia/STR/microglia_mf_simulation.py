import matplotlib.pyplot as plt
import numpy as np
from pyrates import CircuitTemplate, clear
from numba import njit
import pickle

# choose neuron type
neuron_type = 'mg'

# model parameters
C = 50.0
k = 1.0
v_r = -80.0
v_t = -30.0
Delta = 0.3
kappa = 150.0
a = 0.01
b = -20.0
tau_s = 4.0
tau_r = 50.0
tau_d = 500.0
g_i = 2.0
E_i = -60.0
g_e = 10.0
E_e = 0.0
alpha = 0.0
node_vars = {"C": C, "k": k, "v_r": v_r, "v_t": v_t, "tau_u": 1/a, "b": b, "kappa": kappa, "g_i": g_i, "E_i": E_i,
             "g_e": g_e, "E_e": E_e, "tau_s": tau_s, "v": v_t, "Delta": Delta, "tau_r": tau_r,
             "tau_d": tau_d, "alpha": alpha}

edge_vars = [(f'p/{neuron_type}_op/s', f'p/{neuron_type}_op/s_i', {'weight': 1.0})]

# load model template
template = CircuitTemplate.from_yaml(f'config/mf/{neuron_type}')

# update template parameters
template.update_var(node_vars={f"p/{neuron_type}_op/{key}": val for key, val in node_vars.items()}, edge_vars=edge_vars)

# set pyrates-specific parameters
cutoff = 500.0
T = 2000.0 + cutoff
start = 500.0 + cutoff
stop = 1500.0 + cutoff
dt = 1e-2
dts = 1e-1
inp = np.zeros((int(T/dt),)) + 0.5
inp[int(start/dt):int(stop/dt)] += 1.0
backend = 'default'
solver = 'euler'
out_vars = ['r', 'v', 'x']
kwargs = {'vectorize': False, 'float_precision': 'float64', 'decorator': njit, 'fastmath': False}

# perform simulation
results = template.run(simulation_time=T, step_size=dt, backend=backend, solver=solver, sampling_step_size=dts,
                       inputs={f'p/{neuron_type}_op/s_e': inp},
                       outputs={v: f'p/{neuron_type}_op/{v}' for v in out_vars}, cutoff=cutoff)
clear(template)

# save results
pickle.dump({'results': results}, open("results/mf_mg_ablated.pkl", "wb"))

# plot simulated signal
fig, axes = plt.subplots(figsize=(10, 2*len(out_vars)), nrows=len(out_vars))
for i, v in enumerate(out_vars):
    ax = axes[i]
    ax.plot(results[v])
    ax.set_xlabel('time (ms)')
    ax.set_ylabel(v)
plt.show()
