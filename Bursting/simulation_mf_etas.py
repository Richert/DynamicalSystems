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
eta = 0.0  # unit: pA
Delta = 1.0
kappa = 0.1
tau_u = 35.0
b = -2.0
tau_s = 6.0
tau_x = 300.0
g = 15.0
E_r = 0.0

# define inputs
T = 3500.0
cutoff = 500.0
dt = 1e-2
dts = 1e-1
inp = np.zeros((int(T/dt),)) + 33.0
# inp[:int(cutoff/dt)] -= 15.0
inp[int(1000/dt):int(2000/dt),] += 15.8

# run the model
###############

# initialize model
ik = CircuitTemplate.from_yaml("config/mf/recovery")

# update parameters
ik.update_var(node_vars={'p/recovery_op/C': C, 'p/recovery_op/k': k, 'p/recovery_op/v_r': v_r, 'p/recovery_op/v_t': v_t,
                         'p/recovery_op/Delta': Delta, 'p/recovery_op/kappa': kappa, 'p/recovery_op/tau_u': tau_u,
                         'p/recovery_op/b': b, 'p/recovery_op/tau_s': tau_s, 'p/recovery_op/g': g,
                         'p/recovery_op/E_r': E_r, 'p/recovery_op/tau_x': tau_x, 'p/recovery_op/eta': eta})

# run simulation
res = ik.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='euler',
             outputs={'s': 'p/recovery_op/s', 'u': 'p/recovery_op/u'}, inputs={'p/recovery_op/I_ext': inp},
             decorator=nb.njit, fastmath=True, float_precision="float64")

# plot results
fig, ax = plt.subplots(nrows=2, figsize=(12, 5))
ax[0].plot(res["s"])
ax[0].set_ylabel(r'$s(t)$')
ax[1].plot(res["u"])
ax[1].set_ylabel(r'$u(t)$')
ax[1].set_xlabel("time (ms)")
plt.tight_layout()
plt.show()
