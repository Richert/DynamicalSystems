from pyrates.utility.grid_search import grid_search
from pyrates.utility.data_analysis import welch
import numpy as np
import pickle
from matplotlib.pyplot import show
import pandas as pd

# config
########

# parameters
dt = 1e-3
dts = 1e-1
T = 61000.0
cutoff = 1000.0
sim_steps = int(np.round(T/dt))

# model parameters
k_gp = 7.0
k_p = 2.0
k_i = 1.5
k = 21.2
eta = 20.0
delta = 39.2
param_grid = {
        'k_ee': [0.8*k],
        'k_pe': [8.0*k],
        'k_ep': [3.0*k],
        'k_pp': [0.6*k_gp*k_p*k/k_i],
        'k_ps': [20.0*k],
        'eta_e': [2.5*eta],
        'eta_p': [3.0*eta],
        'eta_s': [0.002],
        'delta_e': [0.3*delta],
        'delta_p': [0.9*delta],
        'tau_e': [19.9],
        'tau_p': [37.1],
        'tau_ampa_r': [0.7],
        'tau_ampa_d': [5.6],
        'tau_gabaa_r': [1.3],
        'tau_gabaa_d': [15.8],
        'tau_stn': [1.5]
    }
param_grid = pd.DataFrame.from_dict(param_grid)

param_map = {
    'k_ee': {'vars': ['weight'], 'edges': [('stn', 'stn')]},
    'k_pe': {'vars': ['weight'], 'edges': [('stn', 'gpe_p')]},
    'k_pp': {'vars': ['weight'], 'edges': [('gpe_p', 'gpe_p')]},
    'k_ep': {'vars': ['weight'], 'edges': [('gpe_p', 'stn')]},
    'k_ps': {'vars': ['weight'], 'edges': [('str', 'gpe_p')]},
    'eta_e': {'vars': ['stn_syns_op/eta_e'], 'nodes': ['stn']},
    'eta_p': {'vars': ['gpe_proto_syns_op/eta_i'], 'nodes': ['gpe_p']},
    'eta_s': {'vars': ['str_dummy_op/eta_s'], 'nodes': ['str']},
    'delta_e': {'vars': ['stn_syns_op/delta_e'], 'nodes': ['stn']},
    'delta_p': {'vars': ['gpe_proto_syns_op/delta_i'], 'nodes': ['gpe_p']},
    'tau_e': {'vars': ['stn_syns_op/tau_e'], 'nodes': ['stn']},
    'tau_p': {'vars': ['gpe_proto_syns_op/tau_i'], 'nodes': ['gpe_p']},
    'tau_ampa_d': {'vars': ['gpe_proto_syns_op/tau_ampa_d', 'stn_syns_op/tau_ampa_d'], 'nodes': ['gpe_p', 'stn']},
    'tau_gabaa_d': {'vars': ['gpe_proto_syns_op/tau_gabaa_d', 'stn_syns_op/tau_gabaa_d'], 'nodes': ['gpe_p', 'stn']},
    'tau_ampa_r': {'vars': ['gpe_proto_syns_op/tau_ampa_d', 'stn_syns_op/tau_ampa_d'], 'nodes': ['gpe_p', 'stn']},
    'tau_gabaa_r': {'vars': ['gpe_proto_syns_op/tau_gabaa_d', 'stn_syns_op/tau_gabaa_d'], 'nodes': ['gpe_p', 'stn']},
    'tau_stn': {'vars': ['stn_syns_op/tau_gabaa'], 'nodes': ['stn']}
}

# simulations
#############
results, result_map = grid_search(
    circuit_template="/home/rgast/PycharmProjects/BrainNetworks/BasalGanglia/config/stn_gpe/stn_gpe_2pop",
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
pickle.dump({'results': results, 'psds': psds}, open("results/stn_gpe_2pop_beta_sims.p", "wb"))
