import pandas as pd
import matplotlib.pyplot as plt
from pyrates.utility.grid_search import grid_search
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
delays = [1.0, 2.0, 3.0, 4.0]
spreads = [0.6, 0.8, 1.0, 1.155]
n = len(delays)
param_grid = {
        'k_ae': [k*1.5]*n,
        'k_pe': [k*5.0]*n,
        'k_pp': [5.0*k*k_gp]*n,
        'k_ap': [2.0*k*k_gp]*n,
        'k_aa': [0.1*k*k_gp]*n,
        'k_pa': [0.5*k*k_gp]*n,
        'k_ps': [10.0*k*k_gp]*n,
        'k_as': [1.0*k*k_gp]*n,
        'eta_e': [0.02]*n,
        'eta_p': [30.0]*n,
        'eta_a': [26.0]*n,
        'eta_s': [0.002]*n,
        'delta_p': [9.0]*n,
        'delta_a': [3.0]*n,
        'tau_p': [18]*n,
        'tau_a': [32]*n,
        'delays': delays,
        'spreads': spreads
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
    'tau_a': {'vars': ['gpe_arky_syns_op/tau_a'], 'nodes': ['gpe_a']},
    'delays': {'vars': ['delay'], 'edges': [('gpe_p', 'gpe_p'), ('gpe_p', 'gpe_a'), ('gpe_a', 'gpe_p'),
                                            ('gpe_a', 'gpe_a')]},
    'spreads': {'vars': ['spread'], 'edges': [('gpe_p', 'gpe_p'), ('gpe_p', 'gpe_a'), ('gpe_a', 'gpe_p'),
                                              ('gpe_a', 'gpe_a')]}
}

# simulations
#############

results_col = []
for i in range(n):
    param_grid_tmp = pd.DataFrame.from_dict({key: [val[i]] for key, val in param_grid.items()})
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

    results_col.append(results)
    # fig2, ax = plt.subplots(figsize=(6, 2.0), dpi=dpi)
    # results = results * 1e3
    # plot_timeseries(results, ax=ax)
    # plt.legend(['GPe-p', 'GPe-a'])
    # ax.set_ylabel('Firing rate')
    # ax.set_xlabel('time (ms)')
    # # ax.set_xlim([000.0, 1500.0])
    # # ax.set_ylim([0.0, 100.0])
    # ax.tick_params(axis='both', which='major', labelsize=9)
    # plt.tight_layout()
    # plt.show()

# save results
pickle.dump({f"d{i}": r for i, r in enumerate(results_col)}, open("results/gpe_2pop_syns.p", "wb"))
