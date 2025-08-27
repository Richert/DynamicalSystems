from pyrates import CircuitTemplate
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import numba as nb
plt.rcParams['backend'] = 'TkAgg'

# define parameters
###################

# model parameters
tau = 4.059
Delta = 5.143
eta = 1.528
alpha = 0.511
tau_a = 28.263
A0 = 0.0
J = 45.03
kappa = 0.469
tau_s = 3.095
tau_u = 417.346
I_ext = 0.0
noise_lvl = 10.0
noise_sigma = 1000.0

params = {
    'tau': tau, 'Delta': Delta, 'eta': eta, 'alpha': alpha, 'tau_a': tau_a, 'J': J, 'tau_s': tau_s,
    'tau_u': tau_u, 'kappa': kappa, 'A0': A0
}

# define inputs
cutoff = 0.0
T = 5000.0 + cutoff
dt = 5e-3
dts = 1e-1
start = 500.0 + cutoff
stop = 1500.0 + cutoff
inp = np.zeros((int(T/dt),))
noise = noise_lvl*np.random.randn(inp.shape[0])
noise = gaussian_filter1d(noise, sigma=noise_sigma)
inp += noise
inp[int(start/dt):int(stop/dt)] += I_ext

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
                inputs={f'p/{op}/I_ext': inp}, clear=False)

# plot results
fig, axes = plt.subplots(nrows=3, figsize=(12, 6), sharex=True)
fig.suptitle("Mean-field dynamics")
ax = axes[0]
ax.plot(res_mf.index, res_mf["r"])
ax.set_ylabel(r'$r(t)$')
ax.set_xlabel("time (ms)")
ax.set_title("average firing rate")
ax = axes[1]
ax.plot(res_mf.index, res_mf["a"])
ax.set_ylabel(r'$a(t)$')
ax.set_xlabel("time (ms)")
ax.set_title("Synaptic efficacy")
ax = axes[2]
ax.plot(res_mf.index, res_mf["u"])
ax.set_ylabel(r'$u(t)$')
ax.set_xlabel("time (ms)")
ax.set_title("SFA")
plt.tight_layout()
plt.show()

# save results
# pickle.dump({'mf': res_mf, "snn": res_snn}, open("results/rs_high_sfa.p", "wb"))
