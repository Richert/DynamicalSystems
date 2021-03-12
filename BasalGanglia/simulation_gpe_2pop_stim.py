import pandas as pd
import numpy as np
from pyrates.utility.grid_search import grid_search
import pickle
from scipy.ndimage.filters import gaussian_filter1d
from matplotlib.pyplot import show

# parameter definitions
#######################

# simulation parameters
dt = 1e-3
dts = 1e-1
T = 3000.0

# stimulation parameters
sim_steps = int(np.round(T/dt))
stim_offset = int(np.round(2200.0/dt))
stim_dur = int(np.round(400.0/dt))
stim_delayed = int(np.round((2600.0)/dt))
stim_amp = 10.0
stim_var = 10.0

ctx = np.zeros((sim_steps, 1))
ctx[stim_offset:stim_offset+stim_dur, 0] = stim_amp
# ctx[stim_delayed:stim_delayed+stim_dur, 0] = -stim_amp
ctx = gaussian_filter1d(ctx, stim_var, axis=0)

# model parameters
k_gp = 1.0
k = 10.0
param_grid = {
        'k_ae': [k*1.5],
        'k_pe': [k*5.0],
        'k_pp': [5.0*k*k_gp],
        'k_ap': [2.0*k*k_gp],
        'k_aa': [0.1*k*k_gp],
        'k_pa': [0.5*k*k_gp],
        'k_ps': [10.0*k*k_gp],
        'k_as': [1.0*k*k_gp],
        'eta_e': [0.02],
        'eta_p': [30.0],
        'eta_a': [26.0],
        'eta_s': [0.002],
        'delta_p': [9.0],
        'delta_a': [3.0],
        'tau_p': [18],
        'tau_a': [32],
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

results, result_map = grid_search(
        circuit_template="/home/rgast/PycharmProjects/BrainNetworks/BasalGanglia/config/stn_gpe/gpe_2pop",
        param_grid=param_grid,
        param_map=param_map,
        simulation_time=T,
        step_size=dt,
        permute=True,
        sampling_step_size=dts,
        inputs={
            'gpe_p/gpe_proto_syns_op/I_ext': ctx,
            },
        outputs={
            'r_i': 'gpe_p/gpe_proto_syns_op/R_i',
            'r_a': 'gpe_a/gpe_arky_syns_op/R_a',
        },
        init_kwargs={
            'backend': 'numpy', 'solver': 'scipy', 'step_size': dt},
        method='RK45',
    )

results.plot()
show()

# save results
pickle.dump({'results': results}, open("results/gpe_2pop_stim1.p", "wb"))
