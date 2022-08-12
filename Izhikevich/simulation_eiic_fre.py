"""
This script uses the dynamical systems software PyRates (https://github.com/pyrates-neuroscience/PyRates) to simulate
the dynamics of a three-population model of interconnected Izhikevich neuron populations, representing regular-spiking,
fast-spiking, and low-threshold-spiking neurons. The model equations and default parameters are defined in a separate
YAML file. To run this script, you need to execute it from within a Python >= 3.6 environment with the following
packages installed:
- PyRates (see https://github.com/pyrates-neuroscience/PyRates for installation instructions)
- matplotlib
- numba
Also, make sure that the `path_to_yaml_config` variable provides the correct path to the YAML config file with the
model equations.
"""

from pyrates import CircuitTemplate
import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.rcParams['backend'] = 'TkAgg'
import numba as nb

# define parameters
###################

# model parameters
Delta_rs = 0.5
Delta_fs = 0.8
Delta_lts = 0.1

# define inputs
T = 4000.0
cutoff = 1000.0
start = 500.0
dur = 2000.0
dt = 1e-3
dts = 1e-1
I_r = np.zeros((int(T/dt),)) + 60.0
I_f = np.zeros((int(T/dt),)) + 40.0
I_l = np.zeros((int(T/dt),)) + 80.0
I_l[int((cutoff+start)/dt):int((cutoff+start+dur)/dt)] += 25.0
I_l[int((cutoff+start+0.5*dur)/dt):int((cutoff+start+dur)/dt)] += 25.0

# run the model
###############

# initialize model
path_to_yaml_config = "config/ik/eiic"
eic = CircuitTemplate.from_yaml(path_to_yaml_config)

# update parameters (can also be used to alter the initial state of the system)
eic.update_var(node_vars={'rs/rs_op/Delta': Delta_rs, 'fs/fs_op/Delta': Delta_fs, 'lts/lts_op/Delta': Delta_lts,
                          'rs/rs_op/r': 0.02, 'rs/rs_op/v': -45.0})

# run simulation
res = eic.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='euler',
              outputs={'rs': 'rs/rs_op/r', 'fs': 'fs/fs_op/r', 'lts': 'lts/lts_op/r'},
              inputs={'rs/rs_op/I_ext': I_r, 'fs/fs_op/I_ext': I_f, 'lts/lts_op/I_ext': I_l},
              decorator=nb.njit, fastmath=True, vectorize=False)

# plot results
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(res*1e3)
ax.set_ylabel(r'$r(t)$')
ax.set_xlabel("time (ms)")
plt.legend(res.columns.values)
plt.tight_layout()
plt.show()

# save results
# pickle.dump({'results': res, 'input': I_l}, open("results/eiic_fre_het.p", "wb"))
