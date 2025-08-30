from pyrates import CircuitTemplate
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import numba as nb
plt.rcParams['backend'] = 'TkAgg'

# define parameters
###################

# model parameters
C = 50.0
k = 0.8
v_r = -60.0
v_t = -40.0
Delta = 2.0
eta = 0.0
kappa = 25.0
alpha = 0.9
tau_a = 130.0
tau_x = 1500.0
tau_s = 70.0
A0 = 0.0
g = 220.0
E_r = 0.0
I_ext = 56.0
noise_lvl = 10.0
noise_sigma = 100.0

params = {
    'C': C, 'k': k, 'v_r': v_r, 'v_t': v_t, 'Delta': Delta, 'eta': eta, 'kappa': kappa, 'alpha': alpha,
    'tau_a': tau_a, 'g': g, 'E_r': E_r, 'tau_x': tau_x, 'A0': A0, 'tau_s': tau_s
}

# define inputs
T = 6000.0
cutoff = 0.0
dt = 1e-1
dts = 1.0
inp = np.zeros((int(T/dt),)) + I_ext
# noise = noise_lvl*np.random.randn(inp.shape[0])
# noise = gaussian_filter1d(noise, sigma=noise_sigma)
# inp += noise

# run the mean-field model
##########################

# initialize model
op = "ik_ca_op"
ik = CircuitTemplate.from_yaml("config/ik_mf/ik_ca")

# update parameters
ik.update_var(node_vars={f"p/{op}/{var}": val for var, val in params.items()})

# run simulation
res_mf = ik.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='heun',
                outputs={'r': f'p/{op}/r', 'a': f'p/{op}/a', 'u': f'p/{op}/x'},
                inputs={f'p/{op}/I_ext': inp}, clear=True)

# plot results
fig, axes = plt.subplots(nrows=3, figsize=(12, 8), sharex=True)
fig.suptitle("Mean-field dynamics")
ax = axes[0]
ax.plot(res_mf.index, res_mf["r"]*1e3)
ax.set_ylabel(r'$r(t)$ (Hz)')
ax.set_xlabel("time (ms)")
ax.set_title("average firing rate")
ax = axes[1]
ax.plot(res_mf.index, res_mf["u"])
ax.set_ylabel(r'$u(t)$ (mV)')
ax.set_xlabel("time (ms)")
ax.set_title("Spike frequency adaptation")
ax = axes[2]
ax.plot(res_mf.index, res_mf["a"])
ax.set_ylabel(r'$A(t)$')
ax.set_xlabel("time (ms)")
ax.set_title("Calcium concentration")
plt.tight_layout()
plt.show()

# save results
# pickle.dump({'mf': res_mf, "snn": res_snn}, open("results/rs_high_sfa.p", "wb"))
