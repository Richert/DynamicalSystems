from pyrates import CircuitTemplate
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import numba as nb
plt.rcParams['backend'] = 'TkAgg'

# define parameters
###################

# model parameters
tau = 10.0
Delta = 5.0
eta = -2.0
kappa = 0.0
s = 0.8
theta = 30.0
tau_a = 200.0
J_e = 30.0
I_ext = 2.0
noise_lvl = 50.0
noise_sigma = 100.0

params = {
    'tau': tau, 'Delta': Delta, 'eta': eta, 'kappa': kappa, 's': s, 'theta': theta, 'tau_a': tau_a, 'J_e': J_e
}

# define inputs
T = 10000.0
cutoff = 0.0
dt = 1e-2
dts = 1.0
inp = np.zeros((int(T/dt),)) + I_ext
# noise = noise_lvl*np.random.randn(inp.shape[0])
# noise = gaussian_filter1d(noise, sigma=noise_sigma)
# inp += noise

# run the mean-field model
##########################

# initialize model
op = "qif_ca_op"
ik = CircuitTemplate.from_yaml("config/ik_mf/qif_ca")

# update parameters
ik.update_var(node_vars={f"p/{op}/{var}": val for var, val in params.items()})

# run simulation
res_mf = ik.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='heun',
                outputs={'r': f'p/{op}/r', 'a': f'p/{op}/x', 'v': f'p/{op}/v'},
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
ax.plot(res_mf.index, res_mf["v"])
ax.set_ylabel(r'$v(t)$ (mV)')
ax.set_xlabel("time (ms)")
ax.set_title("Membrane potential")
ax = axes[2]
ax.plot(res_mf.index, res_mf["a"])
ax.set_ylabel(r'$A(t)$')
ax.set_xlabel("time (ms)")
ax.set_title("Calcium concentration")
plt.tight_layout()
plt.show()

# save results
# pickle.dump({'mf': res_mf, "snn": res_snn}, open("results/rs_high_sfa.p", "wb"))
