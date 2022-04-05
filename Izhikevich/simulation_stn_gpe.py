from pyrates import CircuitTemplate
import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.rcParams['backend'] = 'TkAgg'
import numba as nb

# define parameters
###################

# model parameters
Delta_stn = 1.0
Delta_gpe = 2.0
k_stn_gpe = 15.0
k_gpe_stn = 15.0
k_gpe_gpe = 15.0

# define inputs
T = 2500.0
cutoff = 500.0
dt = 1e-3
dts = 1e-1
I_stn = np.zeros((int(T/dt),)) + 70.0
I_gpe = np.zeros((int(T/dt),)) + 60.0
I_stn[int(1000/dt):int(2000/dt)] -= 60.0

# run the model
###############

# initialize model
stn_gpe = CircuitTemplate.from_yaml("config/ik/stn_gpe")

# update parameters
stn_gpe.update_var(node_vars={'stn/stn_op/Delta': Delta_stn, 'gpe/gpe_op/Delta': Delta_gpe},
                   edge_vars=[('stn/stn_op/r', 'gpe/gpe_op/r_e', {'weight': k_stn_gpe}),
                              ('gpe/gpe_op/r', 'stn/stn_op/r_in', {'weight': k_gpe_stn}),
                              ('gpe/gpe_op/r', 'gpe/gpe_op/r_i', {'weight': k_gpe_gpe})])

# generate run function
# stn_gpe.get_run_func(func_name='stn_gpe_run', file_name='stn_gpe', step_size=dt, backend='fortran',
#                      auto=True, vectorize=False, in_place=False)

# run simulation
res = stn_gpe.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='euler',
                  outputs={'stn': 'stn/stn_op/r', 'gpe': 'gpe/gpe_op/r'},
                  inputs={'stn/stn_op/I_ext': I_stn, 'gpe/gpe_op/I_ext': I_gpe},
                  decorator=nb.njit, fastmath=True)

# plot results
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(res*1e3)
ax.set_ylabel(r'$r(t)$')
ax.set_xlabel("time (ms)")
plt.legend(res.columns.values)
plt.tight_layout()
plt.show()

# save results
# pickle.dump({'results': res}, open("results/stn_gpe.p", "wb"))
