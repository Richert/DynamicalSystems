import matplotlib.pyplot as plt
import numpy as np
from pyrates import CircuitTemplate, clear
from numba import njit

# choose neuron type
neuron_type = 'spn'

# redefine model parameters
node_vars = {}
edge_vars = [(f'p/{neuron_type}_op/r', f'p/{neuron_type}_op/r_i', {'weight': 5.0})]

# load model template
template = CircuitTemplate.from_yaml(f'config/model_def/{neuron_type}')

# update template parameters
template.update_var(node_vars=node_vars, edge_vars=edge_vars)

# set pyrates-specific parameters
T = 100.0
start = 300.0
stop = 600.0
dt = 1e-3
dts = 1e-1
inp = np.zeros((int(T/dt),)) + 300.0
inp[int(start/dt):int(stop/dt)] += 0.0
backend = 'default'
solver = 'euler'
out_vars = ['r', 's_gaba', 'v']
kwargs = {'vectorize': False, 'float_precision': 'float64', 'decorator': njit, 'fastmath': False}

# perform simulation
results = template.run(simulation_time=T, step_size=dt, backend=backend, solver=solver, sampling_step_size=dts,
                       inputs={f'p/{neuron_type}_op/I_ext': inp},
                       outputs={v: f'p/{neuron_type}_op/{v}' for v in out_vars})
clear(template)

# plot simulated signal
fig, axes = plt.subplots(figsize=(10, 2*len(out_vars)), nrows=len(out_vars))
for i, v in enumerate(out_vars):
    ax = axes[i]
    ax.plot(results[v])
    ax.set_xlabel('time (ms)')
    ax.set_ylabel(v)
plt.show()
