from pyrates import CircuitTemplate, grid_search
import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.rcParams['backend'] = 'TkAgg'
import numba as nb

# define parameters
###################

# model parameters
C = 100.0   # unit: pF
k = 0.7  # unit: None
v_r = -60.0  # unit: mV
v_t = -40.0  # unit: mV
v_spike = 40.0  # unit: mV
v_reset = 60.0  # unit: mV
Delta = 1.0  # unit: mV
d = 100.0
a = 0.03
b = -2.0
tau_s = 6.0
g = 15.0
E_r = 0.0

# define inputs
T = 2500.0
cutoff = 500.0
dt = 1e-3
dts = 1e-1
inp = np.zeros((int(T/dt),)) + 55.0
inp[int(1000/dt):int(2000/dt)] += 20.0

# run the model
###############

# initialize model
ik = CircuitTemplate.from_yaml("config/ik/ik")

# update parameters
ik.update_var(node_vars={'p/ik_op/C': C, 'p/ik_op/k': k, 'p/ik_op/v_r': v_r, 'p/ik_op/v_t': v_t, 'p/ik_op/v_p': v_spike,
                         'p/ik_op/v_z': v_reset, 'p/ik_op/Delta': Delta, 'p/ik_op/d': d, 'p/ik_op/a': a,
                         'p/ik_op/b': b, 'p/ik_op/tau_s': tau_s, 'p/ik_op/g': g, 'p/ik_op/E_r': E_r})

# run simulation
res = ik.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='euler',
             outputs={'s': 'p/ik_op/s', 'u': 'p/ik_op/u'}, inputs={'p/ik_op/I_ext': inp},
             decorator=nb.njit, fastmath=True)

# plot results
fig, ax = plt.subplots(nrows=2, figsize=(12, 4))
ax[0].plot(res["s"])
ax[0].set_ylabel(r'$s(t)$')
ax[1].plot(res["u"])
ax[1].set_ylabel(r'$u(t)$')
ax[1].set_xlabel("time (ms)")
plt.tight_layout()
plt.show()

# save results
pickle.dump({'results': res}, open("results/rs_fre_het.p", "wb"))
