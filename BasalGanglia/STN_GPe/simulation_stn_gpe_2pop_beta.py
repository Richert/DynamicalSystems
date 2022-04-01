from pyrates.utility.grid_search import grid_search
from pyrates.utility.data_analysis import welch
import numpy as np
import pickle
from matplotlib.pyplot import show
import pandas as pd

# config
########

fid = "stn_gpe_beta"

# parameters
dt = 1e-3
dts = 1e-1
T = 11000.0
cutoff = 1000.0
sim_steps = int(np.round(T/dt))

# model parameters
k = 1000.0
eta = 1000.0
delta = 100.0
k_pe = 6.22
k_ep = 2.69
param_grid = {
        'k_ee': [0.1*k*k_pe],
        'k_pe': [k_pe*k],
        'k_ep': [k_ep*k],
        'k_pp': [1.19*k_ep*k],
        'eta_e': [0.97*eta],
        'eta_p': [-5.36*eta],
        'delta_e': [0.99*delta],
        'delta_p': [2.38*delta],
        'tau_e': [20.87],
        'tau_p': [28.18],
        'tau_ampa_r': [1.12],
        'tau_ampa_d': [9.96],
        'tau_gabaa_r': [1.10],
        'tau_gabaa_d': [19.10],
        'tau_stn': [1.53]
    }
param_grid = pd.DataFrame.from_dict(param_grid)

param_map = {
    'k_ee': {'vars': ['weight'], 'edges': [('stn', 'stn')]},
    'k_pe': {'vars': ['weight'], 'edges': [('stn', 'gpe_p')]},
    'k_pp': {'vars': ['weight'], 'edges': [('gpe_p', 'gpe_p')]},
    'k_ep': {'vars': ['weight'], 'edges': [('gpe_p', 'stn')]},
    'eta_e': {'vars': ['stn_syns_op/eta_e'], 'nodes': ['stn']},
    'eta_p': {'vars': ['gpe_proto_syns_op/eta_i'], 'nodes': ['gpe_p']},
    'delta_e': {'vars': ['stn_syns_op/delta_e'], 'nodes': ['stn']},
    'delta_p': {'vars': ['gpe_proto_syns_op/delta_i'], 'nodes': ['gpe_p']},
    'tau_e': {'vars': ['stn_syns_op/tau_e'], 'nodes': ['stn']},
    'tau_p': {'vars': ['gpe_proto_syns_op/tau_i'], 'nodes': ['gpe_p']},
    'tau_ampa_r': {'vars': ['gpe_proto_syns_op/tau_ampa_r', 'stn_syns_op/tau_ampa_r'], 'nodes': ['gpe_p', 'stn']},
    'tau_ampa_d': {'vars': ['gpe_proto_syns_op/tau_ampa_d', 'stn_syns_op/tau_ampa_d'], 'nodes': ['gpe_p', 'stn']},
    'tau_gabaa_r': {'vars': ['gpe_proto_syns_op/tau_gabaa_r', 'stn_syns_op/tau_gabaa_r'], 'nodes': ['gpe_p', 'stn']},
    'tau_gabaa_d': {'vars': ['gpe_proto_syns_op/tau_gabaa_d', 'stn_syns_op/tau_gabaa_d'], 'nodes': ['gpe_p', 'stn']},
    'tau_stn': {'vars': ['stn_syns_op/tau_gabaa'], 'nodes': ['stn']}
}

# simulations
#############

results, result_map = grid_search(
    circuit_template="../../BrainNetworks/BasalGanglia/config/stn_gpe/stn_gpe_2pop",
    param_grid=param_grid,
    param_map=param_map,
    simulation_time=T,
    step_size=dt,
    permute=True,
    sampling_step_size=dts,
    inputs={
        #'stn/stn_op/ctx': ctx,
        #'str/str_dummy_op/I': stria
        },
    outputs={'r_e': 'stn/stn_syns_op/R_e', 'r_p': 'gpe_p/gpe_proto_syns_op/R_i'},
    init_kwargs={
        'backend': 'numpy', 'solver': 'scipy', 'step_size': dt},
    method='RK45'
)

results = results * 1e3
results = results.loc[cutoff:, :]
results.index = results.index * 1e-3
# results.plot()
# show()

# post-processing
#################

# calculate power-spectral density of firing rate fluctuations
psds = {'freq_stn': [], 'pow_stn': [], 'freq_gpe': [], 'pow_gpe': []}
params = {'k_gp': []}

stn = results.loc[:, 'r_e']
gpe = results.loc[:, 'r_p']

# calculate PSDs
p_stn, f_stn = welch(stn, fmin=2.0, fmax=160.0, n_fft=4096, n_overlap=256)
p_gpe, f_gpe = welch(gpe, fmin=2.0, fmax=160.0, n_fft=4096, n_overlap=256)

# store output quantities
psds['freq_stn'].append(np.squeeze(f_stn))
psds['pow_stn'].append(np.squeeze(p_stn))
psds['freq_gpe'].append(np.squeeze(f_gpe))
psds['pow_gpe'].append(np.squeeze(p_gpe))

# save results
pickle.dump({'results': results, 'psds': psds, 'map': result_map}, open(f"results/{fid}_sims.p", "wb"))
