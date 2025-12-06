from pyrates import CircuitTemplate
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import numba as nb
plt.rcParams['backend'] = 'TkAgg'

# define parameters
###################

# PC parameters
C = 100.0
k = 0.7
v_r = -70.0
v_t = -45.0
Delta = 2.0
eta = 100.0
b = -2.0
kappa = 10.0
U0 = 0.6
alpha = 0.2
g_a = 2.0
g_n = 0.2
g_g = 10.0
tau_w = 100.0
tau_u = 50.0
tau_x = 1000.0
tau_a = 5.0
tau_n = 150.0
tau_g = 10.0
tau_s = 2.0
noise_lvl = 100.0
noise_sigma = 100.0
pc_params = {
    'C': C, 'k': k, 'v_r': v_r, 'v_t': v_t, 'Delta': Delta, 'eta': eta, 'kappa': kappa, 'alpha': alpha,
    'g_a': g_a, 'g_n': g_n, 'g_g': g_g, 'b': b, 'U0': U0, 'tau_w': tau_w, 'tau_u': tau_u, 'tau_x': tau_x, 'tau_a': tau_a,
    'tau_n': tau_n, 'tau_g': tau_g, 'tau_s': tau_s
}

# PV parameters
C = 20.0
k = 1.0
v_r = -60.0
v_t = -40.0
Delta = 2.0
eta = 100.0
b = 0.03
kappa = 10.0
U0 = 1.0
alpha = 0.1
g_a = 2.0
g_n = 0.1
g_g = 5.0
tau_w = 10.0
tau_u = 50.0
tau_x = 500.0
tau_a = 5.0
tau_n = 150.0
tau_g = 10.0
tau_s = 2.0
pv_params = {
    'C': C, 'k': k, 'v_r': v_r, 'v_t': v_t, 'Delta': Delta, 'eta': eta, 'kappa': kappa, 'alpha': alpha,
    'g_a': g_a, 'g_n': g_n, 'g_g': g_g, 'b': b, 'U0': U0, 'tau_w': tau_w, 'tau_u': tau_u, 'tau_x': tau_x, 'tau_a': tau_a,
    'tau_n': tau_n, 'tau_g': tau_g, 'tau_s': tau_s
}

# define inputs
T = 5000.0
cutoff = 0.0
dt = 1e-3
dts = 1.0
inp = np.zeros((int(T/dt),))
noise = noise_lvl*np.random.randn(inp.shape[0])
noise = gaussian_filter1d(noise, sigma=noise_sigma)
inp += noise

# run the mean-field model
##########################

# initialize model
op = "ik_full_op"
ik = CircuitTemplate.from_yaml("config/ik_mf/eic")

# update parameters
ik.update_var(node_vars={f"pc/{op}/{var}": val for var, val in pc_params.items()})
ik.update_var(node_vars={f"pv/{op}/{var}": val for var, val in pv_params.items()})

# run simulation
res_mf = ik.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='scipy',
                outputs={'r_e': f'pc/{op}/r', 'r_i': f'pv/{op}/r',},
                inputs={f'pc/{op}/I_ext': inp}
                )

# plot results
fig, axes = plt.subplots(nrows=2, figsize=(12, 6))
fig.suptitle("Mean-field dynamics")
ax = axes[0]
ax.plot(res_mf.index, res_mf["r_e"] * 1e3)
ax.set_ylabel(r'$r_e(t)$ (Hz)')
ax.set_title("EIC Mean-Field Dynamics")
ax = axes[1]
ax.plot(res_mf.index, res_mf["r_i"] * 1e3)
ax.set_ylabel(r'$r_i(t)$ (Hz)')
ax.set_xlabel("time (ms)")
plt.tight_layout()
plt.show()

# save results
# pickle.dump({'mf': res_mf, "snn": res_snn}, open("results/rs_high_sfa.p", "wb"))
