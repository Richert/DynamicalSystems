import numpy as np
import matplotlib.pyplot as plt
from pyrates import CircuitTemplate
from scipy.ndimage import gaussian_filter1d

# define parameters
###################

# model parameters
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
Delta = 1.1
eta = 50.0
a = 0.03
b = -2.0
d = 100.0
g = 15.0
E_r = 0.0
tau_s = 6.0

# define inputs
T = 3000.0
cutoff = 1000.0
stim = 100.0 + cutoff
amp = 5000.0
sigma = 200
dt = 1e-2
dts = 1e-1
inp = np.zeros((int(T/dt), 1)) + eta
# inp[:int(cutoff*0.5/dt), 0] -= 15.0
inp[int(stim/dt), 0] += amp
inp[:, 0] = gaussian_filter1d(inp[:, 0], sigma=sigma)

# run the mean-field model
##########################

# initialize model
ik = CircuitTemplate.from_yaml("config/ik_mf/ik")

# update parameters
ik.update_var(node_vars={'p/ik_op/C': C, 'p/ik_op/k': k, 'p/ik_op/v_r': v_r, 'p/ik_op/v_t': v_t, 'p/ik_op/Delta': Delta,
                         'p/ik_op/d': d, 'p/ik_op/a': a, 'p/ik_op/b': b, 'p/ik_op/tau_s': tau_s, 'p/ik_op/g': g,
                         'p/ik_op/E_r': E_r})

# run simulation
res_mf = ik.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='scipy', max_step=0.5,
                outputs={'r': 'p/ik_op/r'}, inputs={'p/ik_op/I_ext': inp[:, 0]}, method='LSODA', rtol=1e-8, atol=1e-8)
signal = res_mf["r"]
time = signal.index
fr = signal.values * 1e3

# plotting
##########

fig, axes = plt.subplots(figsize=(12, 6), nrows=2)
ax = axes[0]
ax.plot(time, fr)
ax.set_ylabel("r (Hz)")
ax.set_xlabel("time (ms)")
ax = axes[1]
ax.plot(time, inp[int(cutoff/dt)::int(dts/dt), 0] - eta)
ax.set_ylabel("I_ext (pA)")
ax.set_xlabel("time (ms)")
plt.tight_layout()
plt.show()
