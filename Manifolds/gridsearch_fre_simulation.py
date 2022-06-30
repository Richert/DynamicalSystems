from pyrates import grid_search, CircuitTemplate
import numpy as np
import pickle
import matplotlib.pyplot as plt

# model parameters
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
Delta = 2.0
a = 0.03
b = -2.0
d = 100.0
g = 15.0
E_r = 0.0
tau_s = 6.0

# simulation parameters
T = 2500.0
cutoff = 500.0
dt = 1e-2
dts = 1e-1

# define parameter grid
param_grid = {'eta': [45.0, 60.0, 75.0]}
param_map = {'eta': {'vars': ['ik_op/eta'], 'nodes': ['p']}}

# set up IK mean-field model
ik = CircuitTemplate.from_yaml('config/ik/ik')
ik.update_var(node_vars={'p/ik_op/C': C, 'p/ik_op/k': k, 'p/ik_op/v_r': v_r, 'p/ik_op/v_t': v_t,
                         'p/ik_op/Delta': Delta, 'p/ik_op/a': a, 'p/ik_op/b': b, 'p/ik_op/d': d,
                         'p/ik_op/g': g, 'p/ik_op/E_r': E_r, 'p/ik_op/tau_s': tau_s})

# perform grid-search over etas
results, param_map = grid_search(ik, param_grid=param_grid, param_map=param_map, step_size=dt, simulation_time=T,
                                 sampling_step_size=dts, outputs={'v': 'p/ik_op/v'}, solver='scipy', cutoff=cutoff)

# save results to file
pickle.dump({'results': results, 'map': param_map}, open('results/fre_results.p', 'wb'))

# plot results
plt.plot(results)
plt.show()
