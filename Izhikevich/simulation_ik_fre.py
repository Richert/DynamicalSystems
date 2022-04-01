from pyrates import CircuitTemplate
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
v_spike = 400.0  # unit: mV
v_reset = 600.0  # unit: mV
v_delta = 1.6  # unit: mV
d = 100.0
a = 0.03
b = -2.0
tau_s = 6.0
J = 1.0
g = 20.0
g_e = 0.0
e_r = 0.0

# define inputs
T = 2100.0
cutoff = 100.0
dt = 1e-4
dts = 1e-2
inp = np.zeros((int(T/dt),)) + 60.0
inp[int(600/dt):int(1600/dt)] -= 15.0

# run the model
###############

# initialize model
ik = CircuitTemplate.from_yaml("config/ik/ik")

# update parameters
ik.update_var(node_vars={'p/ik_op/C': C, 'p/ik_op/k': k, 'p/ik_op/v_r': v_r, 'p/ik_op/v_t': v_t, 'p/ik_op/v_p': v_spike,
                         'p/ik_op/v_z': v_reset, 'p/ik_op/Delta': v_delta, 'p/ik_op/d': d, 'p/ik_op/a': a,
                         'p/ik_op/b': b, 'p/ik_op/tau_s': tau_s, 'p/ik_op/g': g, 'p/ik_op/q': g_e, 'p/ik_op/E_r': e_r})

# run simulation
res = ik.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='scipy', method='LSODA',
             outputs={'s': 'p/ik_op/s', 'r': 'p/ik_op/r'}, inputs={'p/ik_op/I_ext': inp}, atol=1e-5, rtol=1e-3,
             decorator=nb.njit)

# plot results
fig, ax = plt.subplots(nrows=2, figsize=(12, 4))
ax[0].plot(res["s"])
ax[0].set_ylabel(r'$s(t)$')
ax[1].plot(res["r"])
ax[1].set_ylabel(r'$r(t)$')
ax[1].set_xlabel("time (ms)")
plt.tight_layout()
plt.show()
