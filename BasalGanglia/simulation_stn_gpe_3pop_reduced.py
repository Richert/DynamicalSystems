from pyrates.utility.grid_search import grid_search
from pyrates.utility.data_analysis import welch
import numpy as np
import pickle
from matplotlib.pyplot import show
from copy import deepcopy

# config
########

# parameters
dt = 1e-3
dts = 1e-1
T = 61000.0
cutoff = 1000.0
sim_steps = int(np.round(T/dt))

# model parameters
k_p = 2.0
k_i = 1.5
k = 100.0
eta = 100.0
param_grid = {
        'k_ee': [0.8*k],
        'k_ae': [3.0*k],
        'k_pe': [8.0*k],
        'k_ep': [10.0*k],
        'k_pp': [1.0*k_p*k/k_i],
        'k_ap': [1.0*k_p*k_i*k],
        'k_aa': [1.0*k/(k_p*k_i)],
        'k_pa': [1.0*k_i*k/k_p],
        'k_ps': [20.0*k],
        'k_as': [20.0*k],
        'eta_e': [4.0*eta],
        'eta_p': [4.0*eta],
        'eta_a': [1.0*eta],
        'eta_s': [0.002],
        'delta_e': [30.0],
        'delta_p': [90.0],
        'delta_a': [120.0],
        'tau_e': [13.0],
        'tau_p': [25.0],
        'tau_a': [20.0],
    }

# parameter sweeps
models = ['stn_gpe_syns', 'stn_gpe_fre', 'stn_gpe_noax', 'stn_gpe_nosyns']
param_updates = [{'k_pa': [0.0], 'k_gp': 10.5, 'eta_p': [3.145*eta]},
                 {'k_gp': 8.0},
                 {'k_gp': 9.8},
                 {'k_gp': 9.3}]
stn_ops = ['stn_syns_op', 'stn_fre_syns_op', 'stn_syns_op', 'stn_op']
gpe_p_ops = ['gpe_proto_syns_op', 'gpe_p_fre_syns_op', 'gpe_proto_syns_op', 'gpe_proto_op']
gpe_a_ops = ['gpe_arky_syns_op', 'gpe_a_fre_syns_op', 'gpe_arky_syns_op', 'gpe_arky_op']

# simulations
#############

results_col = []
for model, params, stn_op, gpe_p_op, gpe_a_op in zip(models, param_updates, stn_ops, gpe_p_ops, gpe_a_ops):

    # adjust parameters
    grid = deepcopy(param_grid)
    k_gp = params.pop('k_gp')
    grid['k_pp'][0] *= k_gp
    grid['k_pa'][0] *= k_gp
    grid['k_ap'][0] *= k_gp
    grid['k_aa'][0] *= k_gp
    grid.update(params)

    # set up parameter map
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
        'eta_e': {'vars': [f'{stn_op}/eta_e'], 'nodes': ['stn']},
        'eta_s': {'vars': ['str_dummy_op/eta_s'], 'nodes': ['str']},
        'eta_p': {'vars': [f'{gpe_p_op}/eta_i'], 'nodes': ['gpe_p']},
        'eta_a': {'vars': [f'{gpe_a_op}/eta_a'], 'nodes': ['gpe_a']},
        'delta_e': {'vars': [f'{stn_op}/delta_e'], 'nodes': ['stn']},
        'delta_p': {'vars': [f'{gpe_p_op}/delta_i'], 'nodes': ['gpe_p']},
        'delta_a': {'vars': [f'{gpe_a_op}/delta_a'], 'nodes': ['gpe_a']},
        'tau_e': {'vars': [f'{stn_op}/tau_e'], 'nodes': ['stn']},
        'tau_p': {'vars': [f'{gpe_p_op}/tau_i'], 'nodes': ['gpe_p']},
        'tau_a': {'vars': [f'{gpe_a_op}/tau_a'], 'nodes': ['gpe_a']},
    }

    # perform simulation
    results, result_map = grid_search(
        circuit_template=f"/home/rgast/PycharmProjects/BrainNetworks/BasalGanglia/config/stn_gpe/{model}",
        param_grid=grid,
        param_map=param_map,
        simulation_time=T,
        step_size=dt,
        permute_grid=False,
        sampling_step_size=dts,
        inputs={},
        outputs={
            'r_e': f'stn/{stn_op}/R_e',
            'r_i': f'gpe_p/{gpe_p_op}/R_i'
        },
        init_kwargs={
            'backend': 'numpy', 'solver': 'scipy', 'step_size': dt},
        method='RK45'
    )
    results = results * 1e3
    results = results.loc[cutoff:, :]
    results.index = results.index * 1e-3
    results = results.droplevel(1, axis=1)
    results = results.droplevel(1, axis=1)
    results_col.append(results)
    # results.plot()
    # show()

# post-processing
#################

# calculate power-spectral density of firing rate fluctuations
psds = {'freq_stn': [], 'pow_stn': [], 'freq_gpe': [], 'pow_gpe': []}
params = {'k_gp': []}

for r in results_col:

    stn = r.loc[:, 'r_e']
    gpe = r.loc[:, 'r_i']

    # calculate PSDs
    p_stn, f_stn = welch(stn, fmin=2.0, fmax=160.0, n_fft=4096, n_overlap=256)
    p_gpe, f_gpe = welch(gpe, fmin=2.0, fmax=160.0, n_fft=4096, n_overlap=256)

    # store output quantities
    psds['freq_stn'].append(np.squeeze(f_stn))
    psds['pow_stn'].append(np.squeeze(p_stn))
    psds['freq_gpe'].append(np.squeeze(f_gpe))
    psds['pow_gpe'].append(np.squeeze(p_gpe))

# save results
pickle.dump({'results': results_col, 'psds': psds}, open("results/stn_gpe_3pop_reductions_sims.p", "wb"))
