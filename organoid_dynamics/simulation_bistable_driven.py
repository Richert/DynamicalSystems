from pyrates import CircuitTemplate
import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.rcParams['backend'] = 'TkAgg'
import numba as nb
import sys
from scipy.ndimage import gaussian_filter1d

# define parameters
###################

# model parameters
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
Delta = 3.5
eta = 0.0
g = 15.0
E_r = 0.0
tau_s = 6.0
noise_lvl = 1000.0
noise_sigma = 300.0

# define inputs
T = 2500.0
cutoff = 500.0
dt = 5e-3
dts = 1e-1
inp = np.zeros((int(T/dt), 1)) + 33.0
inp[int(900/dt):int(1100/dt), 0] += 0.0
noise = noise_lvl*np.random.randn(inp.shape[0],)
noise = gaussian_filter1d(noise, sigma=noise_sigma)
inp[:, 0] += noise

# run the mean-field model
##########################

# initialize model
ik = CircuitTemplate.from_yaml("config/ik_mf/ik")

# update parameters
ik.update_var(node_vars={'p/ik_op/C': C, 'p/ik_op/k': k, 'p/ik_op/v_r': v_r, 'p/ik_op/v_t': v_t, 'p/ik_op/Delta': Delta,
                         'p/ik_op/tau_s': tau_s, 'p/ik_op/g': g, 'p/ik_op/E_r': E_r, 'p/ik_op/v': v_r})

# run simulation
res_mf = ik.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='heun',
                outputs={'s': 'p/ik_op/s'}, inputs={'p/ik_op/I_ext': inp[:, 0]})

print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "sans"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (12, 4)
plt.rcParams['font.size'] = 12.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.5
markersize = 2

# plot results
fig, ax = plt.subplots()
ax.plot(res_mf.index, 1e3*res_mf["s"] / tau_s, label="Mean-Field")
ax.set_ylabel(r'spike rate (Hz)')
ax.set_xlabel("time (ms)")
ax.legend()
plt.tight_layout()
plt.show()

# save results
# pickle.dump({'mf': res_mf, "snn": res_snn}, open("results/rs_high_sfa.p", "wb"))
