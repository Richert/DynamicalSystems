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
eta = -1.7
alpha = 0.01
tau_a = 400.0
J = 20.0
k = 1.0
tau_ampa = 0.2
tau_nmda = 50.0
I_ext = 0.0
noise_lvl = 30.0
noise_sigma = 100.0

params = {
    'tau': tau, 'Delta': Delta, 'eta': eta, 'alpha': alpha, 'tau_a': tau_a, 'J': J,
    'tau_ampa': tau_ampa, 'tau_nmda': tau_nmda, 'k': k
}

# define inputs
T = 1000.0
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
                outputs={'r': f'p/{op}/r', 'a': f'p/{op}/a', 's_nmda': f'p/{op}/s_nmda'},
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
ax.plot(res_mf.index * 10.0, res_mf["s_nmda"])
ax.set_ylabel(r'$s_{nmda}(t)$')
ax.set_xlabel("time (ms)")
ax.set_title("NMDA dynamics")
plt.tight_layout()
plt.show()

# save results
# pickle.dump({'mf': res_mf, "snn": res_snn}, open("results/rs_high_sfa.p", "wb"))
