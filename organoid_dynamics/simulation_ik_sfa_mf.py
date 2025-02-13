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
v_r = -60.0
v_t = -40.0
Delta = 0.1
eta = 0.0
kappa = 8.0
tau_u = 500.0
g = 20.0
E_r = 0.0
tau_s = 6.0
s_ext = 67.0
noise_lvl = 60.0
noise_sigma = 50.0

params = {
    'C': C, 'k': k, 'v_r': v_r, 'v_t': v_t, 'Delta': Delta, 'eta': eta, 'kappa': kappa, 'tau_u': tau_u,
    'g': g, 'E_r': E_r, 'tau_s': tau_s
}

# define inputs
T = 5000.0
cutoff = 0.0
dt = 1e-2
dts = 1e-1
inp = np.zeros((int(T/dt),)) + s_ext
noise = noise_lvl*np.random.randn(inp.shape[0])
noise = gaussian_filter1d(noise, sigma=noise_sigma)
inp += noise

# run the mean-field model
##########################

# initialize model
op = "ik_sfa_op"
ik = CircuitTemplate.from_yaml("config/ik_mf/ik_sfa")

# update parameters
ik.update_var(node_vars={f"p/{op}/{var}": val for var, val in params.items()})

# run simulation
res_mf = ik.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='euler',
                outputs={'r': f'p/{op}/r', 'u': f'p/{op}/u', 'v': f'p/{op}/v'}, inputs={f'p/{op}/I_ext': inp},
                decorator=nb.njit)

# plot results
fig, axes = plt.subplots(nrows=3, figsize=(12, 8))
fig.suptitle("Mean-field dynamics")
ax = axes[0]
ax.plot(res_mf.index, res_mf["r"] * 1e3)
ax.set_ylabel(r'$r(t)$ (Hz)')
ax.set_xlabel("time (ms)")
ax.set_title("average firing rate")
ax = axes[1]
ax.plot(res_mf.index, res_mf["u"])
ax.set_ylabel(r'$u(t)$')
ax.set_xlabel("time (ms)")
ax.set_title("average adaptation")
ax = axes[2]
ax.plot(res_mf.index, res_mf["v"])
ax.set_ylabel(r'$v(t)$ (mV)')
ax.set_xlabel("time (ms)")
ax.set_title("average membrane potential")
plt.tight_layout()
plt.show()

# save results
# pickle.dump({'mf': res_mf, "snn": res_snn}, open("results/rs_high_sfa.p", "wb"))
