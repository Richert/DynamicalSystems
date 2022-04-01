from pyrates.utility.grid_search import grid_search
from pyrates.utility.visualization import create_cmap, plot_connectivity
from pyrates.utility.data_analysis import fft
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# config
########

# parameters
dt = 1e-3
dts = 1e-1
T = 2100.0
cutoff = 100.0
sim_steps = int(np.round(T/dt))

# parameter sweeps
sweep_params = [
    ('tau_e', [6.0, 8.0, 10.0, 12.0, 14.0]),
    ('tau_p', [12.0, 16.0, 20.0, 24.0, 28.0])
]

# model parameters
k_gp = 8.0
k_p = 2.0
k_i = 1.5
k_stn = 1.5
k = 100.0
eta = 100.0
param_grid = {
        'k_ee': [1.0*k],
        'k_ae': [5.0*k/k_stn],
        'k_pe': [5.0*k*k_stn],
        'k_ep': [10.0*k],
        'k_pp': [1.0*k_gp*k_p*k/k_i],
        'k_ap': [1.0*k_gp*k_p*k_i*k],
        'k_aa': [1.0*k_gp*k/(k_p*k_i)],
        'k_pa': [1.0*k_gp*k_i*k/k_p],
        'k_ps': [20.0*k],
        'k_as': [20.0*k],
        'eta_e': [4.0*eta],
        'eta_p': [4.0*eta],
        'eta_a': [0.0*eta],
        'eta_s': [0.002],
        'delta_e': [30.0],
        'delta_p': [90.0],
        'delta_a': [120.0],
        'tau_e': [13],
        'tau_p': [25],
        'tau_a': [20],
    }

param_map = {
    'k_ee': {'vars': ['weight'], 'edges': [('stn', 'stn')]},
    'k_ae': {'vars': ['weight'], 'edges': [('stn', 'gpe_a')]},
    'k_pe': {'vars': ['weight'], 'edges': [('stn', 'gpe_p')]},
    'k_pp': {'vars': ['weight'], 'edges': [('gpe_p', 'gpe_p')]},
    'k_ep': {'vars': ['weight'], 'edges': [('gpe_p', 'stn')]},
    'k_ap': {'vars': ['weight'], 'edges': [('gpe_p', 'gpe_a')]},
    'k_aa': {'vars': ['weight'], 'edges': [('gpe_a', 'gpe_a')]},
    'k_pa': {'vars': ['weight'], 'edges': [('gpe_a', 'gpe_p')]},
    'k_ps': {'vars': ['weight'], 'edges': [('str', 'gpe_p')]},
    'k_as': {'vars': ['weight'], 'edges': [('str', 'gpe_a')]},
    'eta_e': {'vars': ['stn_syns_op/eta_e'], 'nodes': ['stn']},
    'eta_s': {'vars': ['str_dummy_op/eta_s'], 'nodes': ['str']},
    'eta_p': {'vars': ['gpe_proto_syns_op/eta_i'], 'nodes': ['gpe_p']},
    'eta_a': {'vars': ['gpe_arky_syns_op/eta_a'], 'nodes': ['gpe_a']},
    'delta_e': {'vars': ['stn_syns_op/delta_e'], 'nodes': ['stn']},
    'delta_p': {'vars': ['gpe_proto_syns_op/delta_i'], 'nodes': ['gpe_p']},
    'delta_a': {'vars': ['gpe_arky_syns_op/delta_a'], 'nodes': ['gpe_a']},
    'tau_e': {'vars': ['stn_syns_op/tau_e'], 'nodes': ['stn']},
    'tau_p': {'vars': ['gpe_proto_syns_op/tau_i'], 'nodes': ['gpe_p']},
    'tau_a': {'vars': ['gpe_arky_syns_op/tau_a'], 'nodes': ['gpe_a']},
}

# update param grid with sweep params
for key, sweep in sweep_params:
    param_grid[key] = sweep

# grid-search
#############

results, result_map = grid_search(
    circuit_template="/home/rgast/PycharmProjects/BrainNetworks/BasalGanglia/config/stn_gpe/stn_gpe_syns",
    param_grid=param_grid,
    param_map=param_map,
    simulation_time=T,
    step_size=dt,
    permute_grid=True,
    sampling_step_size=dts,
    inputs={},
    outputs={
        #'r_e': 'stn/stn_syns_op/R_e',
        'r_i': 'gpe_p/gpe_proto_syns_op/R_i',
        #'r_a': 'gpe_a/gpe_arky_syns_op/R_a'
    },
    init_kwargs={
        'backend': 'numpy', 'solver': 'scipy', 'step_size': dt},
    method='RK45'
)
results = results * 1e3
results = results.loc[cutoff:, 'r_i']
results.index = results.index * 1e-3
results.plot()

# post-processing
#################

# calculate power-spectral density of firing rate fluctuations
max_freq = np.zeros((len(sweep_params[0][1]), len(sweep_params[1][1])))
max_pow = np.zeros_like(max_freq)

for key in result_map.index:

    val = results.loc[:, key]

    # calculate PSDs
    freqs, power = fft(val)

    # store output quantities
    idx_c = np.argwhere(sweep_params[0][1] == result_map.loc[key, sweep_params[0][0]])[0]
    idx_r = np.argwhere(sweep_params[1][1] == result_map.loc[key, sweep_params[1][0]])[0]
    max_idx = np.argmax(power)
    max_freq[idx_r, idx_c] = freqs[max_idx]
    max_pow[idx_r, idx_c] = power[max_idx]

#################
# visualization #
#################

# PSD profiles
##############

fig, axes = plt.subplots(ncols=2, figsize=(12, 5))

# plot dominating frequency
plot_connectivity(max_freq, xticklabels=np.round(sweep_params[0][1], decimals=1),
                  yticklabels=np.round(sweep_params[1][1], decimals=1), ax=axes[0])
axes[0].set_xlabel(sweep_params[0][0])
axes[0].set_ylabel(sweep_params[1][0])
axes[0].set_title('Dominant Frequency')

# plot power density of dominating frequency
plot_connectivity(max_pow, xticklabels=np.round(sweep_params[0][1], decimals=1),
                  yticklabels=np.round(sweep_params[1][1], decimals=1), ax=axes[1])
axes[1].set_xlabel(sweep_params[0][0])
axes[1].set_ylabel(sweep_params[1][0])
axes[1].set_title('PSD at Dominant Frequency')

plt.tight_layout()
plt.show()