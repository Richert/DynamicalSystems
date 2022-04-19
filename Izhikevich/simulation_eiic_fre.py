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
Delta_fs = 0.5
Delta_lts = 1.6

# define inputs
T = 5000.0
cutoff = 1000.0
dt = 1e-3
dts = 1e-1
I_r = np.zeros((int(T/dt),)) + 60.0
I_f = np.zeros((int(T/dt),)) + 20.0
I_l = np.zeros((int(T/dt),)) + 60.0
I_l[int(2000/dt):int(4000/dt)] += np.linspace(0.0, 100.0, num=int(2000/dt))

# run the model
###############

# initialize model
eic = CircuitTemplate.from_yaml("config/ik/eiic")

# update parameters
eic.update_var(node_vars={'rs/rs_op/Delta': Delta_rs, 'fs/fs_op/Delta': Delta_fs, 'lts/lts_op/Delta': Delta_lts})

# generate run function
# eic.get_run_func(func_name='eic_run', file_name='config/eiic', step_size=dt, backend='fortran',
#                  auto=True, vectorize=False, in_place=False, float_precision='float64', solver='scipy')

# run simulation
res = eic.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='scipy',
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
pickle.dump({'results': res, 'input': I_l}, open("results/eiic_fre_het.p", "wb"))
