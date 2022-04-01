from pyrates.utility.grid_search import grid_search
from pyrates.utility.data_analysis import welch
import numpy as np
import pickle

# config
########

# parameters
dt = 1e-3
dts = 1e-1
T = 11000.0
cutoff = 1000.0
sim_steps = int(np.round(T/dt))

# parameter sweeps
sweep_params = {'k_gp': [5.0, 7.5, 2.8, 11.7],
                'k_pe': [3.2, 8.0, 14.0, 17.0]}

# model parameters
k_gp = 3.0
k_p = 2.0
k_i = 1.5
k = 100.0
eta = 100.0
param_grid = {
        'k_ee': [],
        'k_ae': [],
        'k_pe': [],
        'k_ep': [],
        'k_pp': [],
        'k_ap': [],
        'k_aa': [],
        'k_pa': [],
        'k_ps': [],
        'k_as': [],
        'eta_e': [],
        'eta_p': [],
        'eta_a': [],
        'eta_s': [],
        'delta_e': [],
        'delta_p': [],
        'delta_a': [],
        'tau_e': [],
        'tau_p': [],
        'tau_a': [],
    }

for k_gp, k_pe in zip(sweep_params['k_gp'], sweep_params['k_pe']):

    param_grid['k_ee'].append(0.8*k)
    param_grid['k_ae'].append(3.0*k)
    param_grid['k_pe'].append(k_pe*k)
    param_grid['k_ep'].append(10.0*k)
    param_grid['k_pp'].append(1.0*k_gp*k_p*k/k_i)
    param_grid['k_ap'].append(1.0*k_gp*k_p*k_i*k)
    param_grid['k_aa'].append(1.0*k_gp*k/(k_p*k_i))
    param_grid['k_pa'].append(1.0*k_gp*k_i*k/k_p)
    param_grid['k_ps'].append(20.0*k)
    param_grid['k_as'].append(20.0*k)
    param_grid['eta_e'].append(4*eta)
    param_grid['eta_p'].append(4*eta)
    param_grid['eta_a'].append(1.0)
    param_grid['eta_s'].append(0.002)
    param_grid['delta_e'].append(30.0)
    param_grid['delta_p'].append(90.0)
    param_grid['delta_a'].append(120.0)
    param_grid['tau_e'].append(13.0)
    param_grid['tau_p'].append(25.0)
    param_grid['tau_a'].append(20.0)

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

# grid-search
#############

results, result_map = grid_search(
    circuit_template="/home/rgast/PycharmProjects/BrainNetworks/BasalGanglia/config/stn_gpe/stn_gpe_syns",
    param_grid=param_grid,
    param_map=param_map,
    simulation_time=T,
    step_size=dt,
    permute_grid=False,
    sampling_step_size=dts,
    inputs={},
    outputs={
        'r_e': 'stn/stn_syns_op/R_e',
        'r_i': 'gpe_p/gpe_proto_syns_op/R_i',
        'r_a': 'gpe_a/gpe_arky_syns_op/R_a'
    },
    init_kwargs={
        'backend': 'numpy', 'solver': 'scipy', 'step_size': dt},
    method='RK45'
)
results = results * 1e3
results = results.loc[cutoff:, :]
results.index = results.index * 1e-3
results.plot()

# post-processing
#################

# calculate power-spectral density of firing rate fluctuations
psds = {'freq_stn': [], 'pow_stn': [], 'freq_gpe': [], 'pow_gpe': []}
params = {'k_gp': [], 'k_pe': []}

for key in result_map.index:

    stn = results.loc[:, ('r_e', key)]
    gpe = results.loc[:, ('r_i', key)]
    k_pe = result_map.loc[key, 'k_pe'] / k
    k_gp = result_map.loc[key, 'k_pp'] * k_i/(k_p*k)

    # calculate PSDs
    p_stn, f_stn = welch(stn, fmin=2.0, fmax=160.0, n_fft=4096, n_overlap=256)
    p_gpe, f_gpe = welch(gpe, fmin=2.0, fmax=160.0, n_fft=4096, n_overlap=256)

    # store output quantities
    psds['freq_stn'].append(np.squeeze(f_stn))
    psds['pow_stn'].append(np.squeeze(p_stn))
    psds['freq_gpe'].append(np.squeeze(f_gpe))
    psds['pow_gpe'].append(np.squeeze(p_gpe))
    params['k_pe'].append(k_pe)
    params['k_gp'].append(k_gp)

result_map['k_pe'] = params['k_pe']
result_map['k_gp'] = params['k_gp']

# save results
pickle.dump({'results': results, 'map': result_map, 'psds': psds}, open("results/stn_gpe_3pop_sims.p", "wb"))
