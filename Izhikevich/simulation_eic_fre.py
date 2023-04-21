from pyrates import CircuitTemplate
import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.rcParams['backend'] = 'TkAgg'
import numba as nb
from scipy.ndimage import gaussian_filter1d

# define model parameters
#########################

d_rs = 100.0
Delta_rs = 0.2
Delta_fs = 2.0

# define inputs
T = 2500.0
cutoff = 500.0
dt = 1e-2
dts = 1e-1
I_rs = np.zeros((int(T/dt),)) + 60.0
I_fs = np.zeros((int(T/dt),)) + 43.0
# I_fs[:int(cutoff*0.5/dt)] += 15.0
I_fs[int(750/dt):int(2000/dt)] -= 20.0
I_fs = gaussian_filter1d(I_fs, sigma=1000)

# run the mean-field model
##########################

# initialize model
eic = CircuitTemplate.from_yaml("config/ik_mf/eic")

# update parameters
eic.update_var(node_vars={'rs/rs_op/Delta': Delta_rs, 'fs/fs_op/Delta': Delta_fs, 'rs/rs_op/d': d_rs,
                          'rs/rs_op/v': -40.0, 'rs/rs_op/r': 0.05})

# # generate run function
# eic.get_run_func(func_name='eic_run', file_name='config/eic_shadowing.f90', step_size=dt, backend='fortran',
#                  auto=True, vectorize=False, in_place=False, float_precision='float64', solver='scipy')

# run simulation
res = eic.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='euler',
              outputs={'rs': 'rs/rs_op/r', 'fs': 'fs/fs_op/r'},
              inputs={'rs/rs_op/I_ext': I_rs, 'fs/fs_op/I_ext': I_fs},
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
# pickle.dump({'results': res}, open("results/eic_hom_high_sfa.p", "wb"))
