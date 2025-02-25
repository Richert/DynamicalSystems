from pyrates import CircuitTemplate
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import numba as nb
plt.rcParams['backend'] = 'TkAgg'

# define parameters
###################

# exc parameters
exc_params = {
    'C': 100.0, 'k': 0.7, 'v_r': -60.0, 'v_t': -40.0, 'Delta': 0.5, 'eta': 0.0, 'kappa': 1.6, 'tau_u': 1000.0,
    'g_e': 15.0, 'g_i': 10.0, 'tau_s': 8.0
}

# inh parameters
inh_params = {
    'C': 100.0, 'k': 0.7, 'v_r': -60.0, 'v_t': -40.0, 'Delta': 1.0, 'eta': 40.0, 'g_e': 10.0, 'g_i': 10.0, 'tau_s': 20.0
}

# input parameters
I_e = 200.0
noise_lvl = 20.0
noise_sigma = 50.0

# coupling parameters
g_ee = 1.0
g_ie = 1.0
g_ei = 1.0
g_ii = 1.0

# define inputs
T = 10000.0
cutoff = 0.0
dt = 1e-2
dts = 1.0
inp = np.zeros((int(T/dt),)) + I_e
noise = noise_lvl*np.random.randn(inp.shape[0])
noise = gaussian_filter1d(noise, sigma=noise_sigma)
inp += noise

# run the mean-field model
##########################

# initialize model
exc_op = "ik_sfa_op"
inh_op = "ik_op"
net = CircuitTemplate.from_yaml("config/ik_mf/eic")

# update parameters
net.update_var(node_vars={f"exc/{exc_op}/{var}": val for var, val in exc_params.items()})
net.update_var(node_vars={f"inh/{inh_op}/{var}": val for var, val in inh_params.items()})

# run simulation
res_mf = net.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='heun',
                 outputs={'r_e': f'exc/{exc_op}/r', 'r_i': f'inh/{inh_op}/r'}, inputs={f'exc/{exc_op}/I_ext': inp},
                 decorator=nb.njit)

# plot results
fig, axes = plt.subplots(nrows=2, figsize=(12, 6))
fig.suptitle("Mean-field dynamics")
ax = axes[0]
ax.plot(res_mf.index, res_mf["r_e"] * 1e3)
ax.set_ylabel(r'$r_e(t)$ (Hz)')
ax.set_xlabel("time (ms)")
ax.set_title("excitatory firing rate")
ax = axes[1]
ax.plot(res_mf.index, res_mf["r_i"] * 1e3)
ax.set_ylabel(r'$r_i(t)$')
ax.set_xlabel("time (ms)")
ax.set_title("inhibitory firing rate")
plt.tight_layout()
plt.show()

# save results
# pickle.dump({'mf': res_mf, "snn": res_snn}, open("results/rs_high_sfa.p", "wb"))
