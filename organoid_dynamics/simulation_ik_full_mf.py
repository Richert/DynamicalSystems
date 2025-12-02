from pyrates import CircuitTemplate
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import numba as nb
plt.rcParams['backend'] = 'TkAgg'

# define parameters
###################

# model parameters
C = 100.0
k = 0.7
v_r = -70.0
v_t = -45.0
Delta = 2.0
eta = 0.0
kappa = 200.0
alpha = 0.5
g_a = 10.0
g_n = 5.0
s_ext = 100.0
noise_lvl = 50.0
noise_sigma = 100.0

params = {
    'C': C, 'k': k, 'v_r': v_r, 'v_t': v_t, 'Delta': Delta, 'eta': eta, 'kappa': kappa, 'alpha': alpha,
    'g_a': g_a, 'g_n': g_n
}

# define inputs
T = 2000.0
cutoff = 0.0
dt = 1e-3
dts = 1.0
inp = np.zeros((int(T/dt),)) + s_ext
noise = noise_lvl*np.random.randn(inp.shape[0])
noise = gaussian_filter1d(noise, sigma=noise_sigma)
inp += noise

# run the mean-field model
##########################

# initialize model
op = "ik_full_op"
ik = CircuitTemplate.from_yaml("config/ik_mf/ik_full")

# update parameters
ik.update_var(node_vars={f"p/{op}/{var}": val for var, val in params.items()})

# run simulation
res_mf = ik.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='scipy',
                outputs={'r': f'p/{op}/r', 'w': f'p/{op}/w', 'u': f'p/{op}/u', 'x': f'p/{op}/x'},
                inputs={f'p/{op}/I_ext': inp}
                )

# plot results
fig, axes = plt.subplots(nrows=4, figsize=(12, 8))
fig.suptitle("Mean-field dynamics")
ax = axes[0]
ax.plot(res_mf.index, res_mf["r"] * 1e3)
ax.set_ylabel(r'$r(t)$ (Hz)')
ax.set_title("IK (full model) Mean-Field Dynamics")
ax = axes[1]
ax.plot(res_mf.index, res_mf["w"])
ax.set_ylabel(r'$w(t)$')
ax = axes[2]
ax.plot(res_mf.index, res_mf["u"])
ax.set_ylabel(r'$u(t)$')
ax = axes[3]
ax.plot(res_mf.index, res_mf["x"])
ax.set_ylabel(r'$x(t)$')
ax.set_xlabel("time (ms)")
plt.tight_layout()
plt.show()

# save results
# pickle.dump({'mf': res_mf, "snn": res_snn}, open("results/rs_high_sfa.p", "wb"))
