import matplotlib.pyplot as plt
import numpy as np
from pyrates import CircuitTemplate, clear
from numba import njit

# choose neuron type
neuron_type = 'spn_d2'

# redefine model parameters
node_vars = {f'p/{neuron_type}_op/phi': 1.0}
edge_vars = [(f'p/{neuron_type}_op/r', f'p/{neuron_type}_op/r_i', {'weight': 10.0})]

# load model template
template = CircuitTemplate.from_yaml(f'config/model_def/{neuron_type}')

# update template parameters
template.update_var(node_vars=node_vars, edge_vars=edge_vars)

# set pyrates-specific parameters
T = 1000.0
start = 300.0
stop = 600.0
dt = 1e-3
dts = 1e-1
inp = np.zeros((int(T/dt),)) + 0.01
inp[int(start/dt):int(stop/dt)] += 0.02
backend = 'default'
solver = 'scipy'
out_var = 'r'
kwargs = {'vectorize': False, 'float_precision': 'float64', 'decorator': njit, 'fastmath': False}

# perform simulation
results = template.run(simulation_time=T, step_size=dt, backend=backend, solver=solver, sampling_step_size=dts,
                       inputs={f'p/{neuron_type}_op/r_e': inp}, outputs={out_var: f'p/{neuron_type}_op/{out_var}'})
clear(template)

# plot simulated signal
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(results[out_var])
ax.set_xlabel('time (ms)')
ax.set_ylabel(out_var)
plt.show()
