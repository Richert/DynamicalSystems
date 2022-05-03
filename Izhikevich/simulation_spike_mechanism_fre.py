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
v_spike = 50.0  # unit: mV
v_reset = 80.0  # unit: mV
Delta = 1.0  # unit: mV
d = 0.0
a = 0.03
b = 0.0
tau_s = 6.0
g = 15.0
q = 0.0
E_r = 0.0

# define inputs
T = 2500.0
cutoff = 500.0
dt = 1e-3
dts = 1e-1
inp = np.zeros((int(T/dt),)) + 100.0
inp[int(1000/dt):int(2000/dt)] += 100.0

# run the model
###############

# initialize model
ik = CircuitTemplate.from_yaml("config/ik/ik")

# update parameters
ik.update_var(node_vars={'p/ik_op/C': C, 'p/ik_op/k': k, 'p/ik_op/v_r': v_r, 'p/ik_op/v_t': v_t, 'p/ik_op/v_p': v_spike,
                         'p/ik_op/v_z': v_reset, 'p/ik_op/Delta': Delta, 'p/ik_op/d': d, 'p/ik_op/a': a,
                         'p/ik_op/b': b, 'p/ik_op/tau_s': tau_s, 'p/ik_op/g': g, 'p/ik_op/q': q, 'p/ik_op/E_r': E_r})

# run simulation
res = ik.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='heun',
             outputs={'v': 'p/ik_op/v'}, inputs={'p/ik_op/I_ext': inp},
             decorator=nb.njit, fastmath=True)

# plot results
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(res["v"])
ax.set_ylabel(r'$v(t)$')
ax.set_xlabel("time (ms)")
plt.tight_layout()
plt.show()

# save results
pickle.dump({'results': res}, open("results/spike_mech_fre.p", "wb"))
