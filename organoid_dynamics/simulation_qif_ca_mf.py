from pyrates import CircuitTemplate
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import numba as nb
plt.rcParams['backend'] = 'TkAgg'

# define parameters
###################

# model parameters
tau = 1.0
Delta = 0.5
eta = 0.2
alpha = 0.4
tau_a = 10.0
a_max = 1.0
a_min = 0.0
J = 40.0
kappa = 0.05
tau_s = 2.0
tau_u = 500.0
I_ext = 0.0
noise_lvl = 30.0
noise_sigma = 100.0

params = {
    'tau': tau, 'Delta': Delta, 'eta': eta, 'alpha': alpha, 'tau_a': tau_a, 'J': J, 'tau_s': tau_s,
    'tau_u': tau_u, 'kappa': kappa, 'a_max': a_max, 'a_min': a_min
}

# define inputs
T = 2000.0
cutoff = 100.0
dt = 5e-3
dts = 1e-1
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
                outputs={'r': f'p/{op}/r', 'a': f'p/{op}/a', 'u': f'p/{op}/u'},
                inputs={f'p/{op}/I_ext': inp}, clear=True)

# plot results
fig, axes = plt.subplots(nrows=3, figsize=(12, 8), sharex=True)
fig.suptitle("Mean-field dynamics")
ax = axes[0]
ax.plot(res_mf.index * 10.0, res_mf["r"]*1e2)
ax.set_ylabel(r'$r(t)$ (Hz)')
ax.set_xlabel("time (ms)")
ax.set_title("average firing rate")
ax = axes[1]
ax.plot(res_mf.index * 10.0, res_mf["a"])
ax.set_ylabel(r'$a(t)$')
ax.set_xlabel("time (ms)")
ax.set_title("Synaptic efficacy")
ax = axes[2]
ax.plot(res_mf.index * 10.0, res_mf["u"])
ax.set_ylabel(r'$u(t)$')
ax.set_xlabel("time (ms)")
ax.set_title("SFA")
plt.tight_layout()
plt.show()

# save results
# pickle.dump({'mf': res_mf, "snn": res_snn}, open("results/rs_high_sfa.p", "wb"))
