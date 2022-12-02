from pyrates import CircuitTemplate
import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.rcParams['backend'] = 'TkAgg'
import numba as nb
from scipy.ndimage import gaussian_filter1d

# define parameters
###################

# model parameters
Delta_rs = 1.0
Delta_fs = 0.3

# define inputs
T = 4000.0
cutoff = 1000.0
dt = 1e-3
dts = 1e-1
I_r = np.zeros((int(T/dt),)) + 50.0
I_i = np.zeros((int(T/dt),)) + 20.0
I_i[int(1500/dt):int(3500/dt)] += 10.0
I_i[int(2500/dt):int(3500/dt)] += 10.0
#I_i = gaussian_filter1d(I_i, sigma=3000)

# run the model
###############

# initialize model
eic = CircuitTemplate.from_yaml("config/ik2/eic")

# update parameters
eic.update_var(node_vars={'rs/rs_op/Delta': Delta_rs, 'fs/fs_op/Delta': Delta_fs, 'rs/rs_op/r': 0.0,
                          'rs/rs_op/v': -60.0})

# run simulation
res = eic.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='euler',
              outputs={'rs': 'rs/rs_op/r', 'fs': 'fs/fs_op/r'},
              inputs={'rs/rs_op/I_ext': I_r, 'fs/fs_op/I_ext': I_i},
              decorator=nb.njit, fastmath=True, vectorize=False)

# plot results
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(res*1e3)
ax.set_ylabel(r'$s(t)$')
ax.set_xlabel("time (ms)")
plt.legend(res.columns.v1)
plt.tight_layout()
plt.show()

# save results
pickle.dump({'results': res}, open("results/pole_comp_eic_new.p", "wb"))
