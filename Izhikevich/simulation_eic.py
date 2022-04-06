from pyrates import CircuitTemplate
import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.rcParams['backend'] = 'TkAgg'
import numba as nb

# define parameters
###################

# model parameters
Delta_rs = 1.0
Delta_ib = 2.0

# define inputs
T = 2500.0
cutoff = 500.0
dt = 1e-3
dts = 1e-1
I_r = np.zeros((int(T/dt),)) + 50.0
I_i = np.zeros((int(T/dt),)) + 200
I_i[int(1000/dt):int(2000/dt)] += 100.0

# run the model
###############

# initialize model
eic = CircuitTemplate.from_yaml("config/ik/eic")

# update parameters
eic.update_var(node_vars={'rs/rs_op/Delta': Delta_rs, 'ib/ib_op/Delta': Delta_ib})

# generate run function
# eic.get_run_func(func_name='eic_run', file_name='config/eic', step_size=dt, backend='fortran',
#                  auto=True, vectorize=False, in_place=False, float_precision='float64', solver='scipy')

# run simulation
res = eic.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='scipy',
              outputs={'rs': 'rs/rs_op/v', 'ib': 'ib/ib_op/v'},
              inputs={'rs/rs_op/I_ext': I_r, 'ib/ib_op/I_ext': I_i},
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
pickle.dump({'results': res}, open("results/eic_fre_het.p", "wb"))
