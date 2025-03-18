from pyrates import CircuitTemplate
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from custom_functions import get_bursting_stats
import numba as nb
plt.rcParams['backend'] = 'TkAgg'

# define parameters
###################

# set meta parameters
tau = 10.0
sigma = 100.0
burst_width = 100
burst_sep = 1000
burst_height = 0.5
burst_relheight = 0.9
n_bins = 10
waveform_length = 3000
p_e = 0.8

# exc parameters
exc_params = {
    'C': 100.0, 'k': 0.7, 'v_r': -60.0, 'v_t': -40.0, 'Delta': 2.0, 'eta': 70.0, 'kappa': 10.0, 'tau_u': 500.0,
    'g_e': 150.0, 'g_i': 100.0, 'tau_s': 5.0
}

# inh parameters
inh_params = {
    'C': 100.0, 'k': 0.7, 'v_r': -60.0, 'v_t': -40.0, 'Delta': 4.0, 'eta': 0.0, 'kappa': 20.0, 'tau_u': 200.0,
    'g_e': 60.0, 'g_i': 0.0, 'tau_s': 5.0
}

# input parameters
I_e = 0.0
noise_lvl = 0.0
noise_sigma = 10.0

# define inputs
cutoff = 1000.0
T = 10000.0 + cutoff
dt = 1e-1
dts = 1.0
inp = np.zeros((int(T/dt),)) + I_e
noise = noise_lvl*np.random.randn(inp.shape[0])
noise = gaussian_filter1d(noise, sigma=noise_sigma)
inp += noise

# run the mean-field model
##########################

# initialize model
exc_op = "ik_sfa_op"
inh_op = "ik_sfa_op"
net = CircuitTemplate.from_yaml("config/ik_mf/ik_eic_sfa")

# update parameters
net.update_var(node_vars={f"exc/{exc_op}/{var}": val for var, val in exc_params.items()})
net.update_var(node_vars={f"inh/{inh_op}/{var}": val for var, val in inh_params.items()})

# run simulation
res_mf = net.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='heun',
                 outputs={'r_e': f'exc/{exc_op}/r', 'r_i': f'inh/{inh_op}/r'}, inputs={f'exc/{exc_op}/I_ext': inp},
                 decorator=nb.njit)
res_mf *= 1e3

# get bursting stats
fr = p_e*res_mf["r_e"].values + (1-p_e)*res_mf["r_i"].values
res = get_bursting_stats(fr, sigma=sigma, burst_width=burst_width, rel_burst_height=burst_height,
                         burst_sep=burst_sep, width_at_height=burst_relheight, waveform_length=waveform_length)

# plot results
fig = plt.figure(figsize=(12, 9))
grid = fig.add_gridspec(nrows=2, ncols=2)

# firing rate dynamics
ax = fig.add_subplot(grid[0, :])
l1 =ax.plot(res_mf["r_e"], color="royalblue", label="exc")
ax2 = ax.twinx()
l2 = ax2.plot(res_mf["r_i"], color="darkorange", label="inh")
ax.set_xlabel("time")
ax.set_ylabel("r_e", color="royalblue")
ax2.set_ylabel("r_i", color="darkorange")
ax.set_title("average firing rate dynamics")

# ibi distribution
ax = fig.add_subplot(grid[1, 0])
ax.hist(res["ibi"], n_bins)
ax.set_xlabel("ibi")
ax.set_ylabel("count")
ax.set_title("inter-burst interval distribution")

# waveform
ax = fig.add_subplot(grid[1, 1])
y = res["waveform_mean"]
y_std = res["waveform_std"]
ax.plot(y, color="black")
ax.fill_between(x=np.arange(len(y)), y1=y-y_std, y2=y+y_std, alpha=0.5, color="black")
ax.set_xlabel("time")
ax.set_ylabel("firing rate")
ax.set_title("burst waveform")

plt.tight_layout()
plt.show()
