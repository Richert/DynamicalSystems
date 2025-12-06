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
Delta = 1.0
eta = 87.0
b = -2.0
kappa = 60.0
X0 = 0.0
alpha = 0.0
psi = 500.0
theta = 0.02
g_a = 3.0
g_n = 0.3
g_g = 3.0
tau_u = 60.0
tau_w = 300.0
tau_x = 700.0
tau_a = 5.0
tau_n = 150.0
tau_g = 10.0
params = {
    'C': C, 'k': k, 'v_r': v_r, 'v_t': v_t, 'Delta': Delta, 'eta': eta, 'kappa': kappa, 'alpha': alpha,
    'g_a': g_a, 'g_n': g_n, 'g_g': g_g, 'b': b, 'X0': X0, 'tau_u': tau_u, 'tau_w': tau_w, 'tau_x': tau_x,
    'tau_a': tau_a, 'tau_n': tau_n, 'tau_g': tau_g, 'psi': psi, 'theta': theta
}

# input parameters
s_in = 2.0
noise_lvl = 100.0
noise_sigma = 100.0

# define inputs
T = 10000.0
cutoff = 0.0
dt = 1e-3
dts = 1.0
inp = np.zeros((int(T/dt),))
inp[int(1000/dt):int(2000/dt)] += s_in
noise = noise_lvl*np.random.randn(inp.shape[0])
noise = gaussian_filter1d(noise, sigma=noise_sigma)
inp += noise

# run the mean-field model
##########################

# initialize model
op = "ik_full_op"
ik = CircuitTemplate.from_yaml("config/ik_mf/pc")

# update parameters
ik.update_var(node_vars={f"p/{op}/{var}": val for var, val in params.items()})

# run simulation
res_mf = ik.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='scipy',
                outputs={'r': f'p/{op}/r', 'u': f'p/{op}/u', 'x': f'p/{op}/x'},
                inputs={f'p/{op}/I_ext': inp}
                )

# plot results
fig, axes = plt.subplots(nrows=3, figsize=(12, 6))
fig.suptitle("Mean-field dynamics")
ax = axes[0]
ax.plot(res_mf.index, res_mf["r"] * 1e3)
ax.set_ylabel(r'$r(t)$ (Hz)')
ax.set_title("IK (full model) Mean-Field Dynamics")
ax = axes[1]
ax.plot(res_mf.index, res_mf["u"])
ax.set_ylabel(r'$u(t)$')
ax = axes[2]
ax.plot(res_mf.index, res_mf["x"])
ax.set_ylabel(r'$x(t)$')
ax.set_xlabel("time (ms)")
plt.tight_layout()
plt.show()

# save results
# pickle.dump({'mf': res_mf, "snn": res_snn}, open("results/rs_high_sfa.p", "wb"))
