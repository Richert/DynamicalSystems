from pyrates import CircuitTemplate
import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.rcParams['backend'] = 'TkAgg'
import numba as nb

# define parameters
###################

# model parameters
C = 68.0   # unit: pF
k = 0.94  # unit: None
v_r = -53.0  # unit: mV
v_t = -44.0  # unit: mV
v_spike = 30.0  # unit: mV
v_reset = 60.0  # unit: mV
Delta = 0.2  # unit: mV
d = 0.35
a = 0.005
b = 3.9
tau_s = 6.0
J = 1.0
g = 15.0
q = 0.0
E_r = -60.0

# define inputs
T = 2500.0
cutoff = 500.0
dt = 1e-3
dts = 1e-1
inp = np.zeros((int(T/dt),)) + 40.0
inp[int(1000/dt):int(2000/dt)] += 20.0

# run the model
###############

# initialize model
ik = CircuitTemplate.from_yaml("config/ik/ik2")

# update parameters
ik.update_var(node_vars={'p/ik2_op/C': C, 'p/ik2_op/k': k, 'p/ik2_op/v_r': v_r, 'p/ik2_op/v_t': v_t,
                         'p/ik2_op/v_p': v_spike, 'p/ik2_op/v_z': v_reset, 'p/ik2_op/Delta': Delta,
                         'p/ik2_op/d': d, 'p/ik2_op/a': a, 'p/ik2_op/b': b, 'p/ik2_op/tau_s': tau_s, 'p/ik2_op/g': g,
                         'p/ik2_op/q': q, 'p/ik2_op/E_r': E_r})

# run simulation
res = ik.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='euler',
             outputs={'s': 'p/ik2_op/s', 'u': 'p/ik2_op/u'}, inputs={'p/ik2_op/I_ext': inp},
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
pickle.dump({'results': res}, open("results/ik2_fre_inh_hom.p", "wb"))
