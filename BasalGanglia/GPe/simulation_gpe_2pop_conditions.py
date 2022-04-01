import pandas as pd
import numpy as np
from pyrates.utility.grid_search import grid_search
from copy import deepcopy
import pickle

# parameter definitions
#######################

# simulation parameters
dt = 1e-3
dts = 1e-1
T = 2000.0

# model parameters
k_gp = 1.0
k = 10.0
param_grid = {
        'k_ae': [k*1.5],
        'k_pe': [k*5.0],
        'k_pp': [1.5*k*k_gp],
        'k_ap': [2.0*k*k_gp],
        'k_aa': [0.1*k*k_gp],
        'k_pa': [0.5*k*k_gp],
        'k_ps': [10.0*k*k_gp],
        'k_as': [1.0*k*k_gp],
        'eta_e': [0.02],
        'eta_p': [12.0],
        'eta_a': [26.0],
        'eta_s': [0.002],
        'delta_p': [9.0],
        'delta_a': [3.0],
        'tau_p': [18],
        'tau_a': [32],
        #'omega': stim_periods,
        #'alpha': np.asarray(stim_amps)
    }
param_grid = pd.DataFrame.from_dict(param_grid)

param_map = {
    'k_ae': {'vars': ['weight'], 'edges': [('stn', 'gpe_a')]},
    'k_pe': {'vars': ['weight'], 'edges': [('stn', 'gpe_p')]},
    'k_pp': {'vars': ['weight'], 'edges': [('gpe_p', 'gpe_p')]},
    'k_ap': {'vars': ['weight'], 'edges': [('gpe_p', 'gpe_a')]},
    'k_aa': {'vars': ['weight'], 'edges': [('gpe_a', 'gpe_a')]},
    'k_pa': {'vars': ['weight'], 'edges': [('gpe_a', 'gpe_p')]},
    'k_ps': {'vars': ['weight'], 'edges': [('str', 'gpe_p')]},
    'k_as': {'vars': ['weight'], 'edges': [('str', 'gpe_a')]},
    'eta_p': {'vars': ['gpe_proto_syns_op/eta_i'], 'nodes': ['gpe_p']},
    'eta_a': {'vars': ['gpe_arky_syns_op/eta_a'], 'nodes': ['gpe_a']},
    'eta_e': {'vars': ['stn_dummy_op/eta_e'], 'nodes': ['stn']},
    'eta_s': {'vars': ['str_dummy_op/eta_s'], 'nodes': ['str']},
    'delta_p': {'vars': ['gpe_proto_syns_op/delta_i'], 'nodes': ['gpe_p']},
    'delta_a': {'vars': ['gpe_arky_syns_op/delta_a'], 'nodes': ['gpe_a']},
    'tau_p': {'vars': ['gpe_proto_syns_op/tau_i'], 'nodes': ['gpe_p']},
    'tau_a': {'vars': ['gpe_arky_syns_op/tau_a'], 'nodes': ['gpe_a']}
}

conditions = [{},  # healthy control -> GPe-p: 60 Hz, GPe-a: 10 Hz
              {'eta_s': 20.0},  # STR excitation -> GPe-p: 10 Hz, GPe-a: 40 Hz
              {'eta_e': 0.1},  # STN inhibition -> GPe-p: 30 Hz, GPe_a: 20 Hz
              {'k_pp': 0.1, 'k_pa': 0.1, 'k_aa': 0.1, 'k_ap': 0.1, 'k_ps': 0.1,
               'k_as': 0.1},  # GABAA blockade in GPe -> GPe_p: 100 Hz
              {'k_pe': 0.1, 'k_pp': 0.1, 'k_pa': 0.1, 'k_ae': 0.1, 'k_aa': 0.1, 'k_ap': 0.1,
               'k_ps': 0.1, 'k_as': 0.1},  # AMPA blockade and GABAA blockade in GPe -> GPe_p: 70 Hz
              ]
condition_keys = ['control', 'STR +', 'STN -', 'GABAA -', 'AMPA/GABAA -']

# simulations
#############

rates = pd.DataFrame(data=np.asarray([[0.0, 'dummy', 'pop'] for _ in range(2*len(conditions))]),
                     columns=['r', 'condition', 'population'])
for i, c_dict in enumerate(deepcopy(conditions)):
    for key in param_grid:
        if key in c_dict:
            c_dict[key] = np.asarray(param_grid[key]) * c_dict[key]
        elif key in param_grid:
            c_dict[key] = np.asarray(param_grid[key])
    param_grid_tmp = pd.DataFrame.from_dict(c_dict)
    results, result_map = grid_search(
        circuit_template="/home/rgast/PycharmProjects/BrainNetworks/BasalGanglia/config/stn_gpe/gpe_2pop",
        param_grid=param_grid_tmp,
        param_map=param_map,
        simulation_time=T,
        step_size=dt,
        permute=True,
        sampling_step_size=dts,
        inputs={},
        outputs={
            'r_i': 'gpe_p/gpe_proto_syns_op/R_i',
            'r_a': 'gpe_a/gpe_arky_syns_op/R_a',
        },
        init_kwargs={
            'backend': 'numpy', 'solver': 'scipy', 'step_size': dt},
        method='RK45',
    )

    r_p, r_a = results['r_i'], results['r_a']
    rates.loc[2*i, :] = [r_p.iloc[-1, 0]*1e3, condition_keys[i], 'GPe-p']
    rates.loc[2*i+1, :] = [r_a.iloc[-1, 0]*1e3, condition_keys[i], 'GPe-a']

# save results
pickle.dump({'rates': rates}, open("results/gpe_2pop_config.p", "wb"))
