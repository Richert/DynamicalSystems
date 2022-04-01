from pyrates.utility.grid_search import grid_search
import numpy as np
import pickle
import os
import pandas as pd
import h5py
from matplotlib.pyplot import show

# find fittest candidate among fitting results
##############################################

fid = "stn_gpe_nobeta"
directory = f"/home/rgast/JuliaProjects/JuRates/BasalGanglia/results/{fid}_results"
dv = 'f'

# load data into frame
df = pd.DataFrame(data=np.zeros((1, 1)), columns=["fitness"])
for fn in os.listdir(directory):
    if fn.startswith(fid) and fn.endswith(".h5"):
        f = h5py.File(f"{directory}/{fn}", 'r')
        index = int(fn.split('_')[-1][:-3])
        df_tmp = pd.DataFrame(data=np.asarray([1/f["f/f"][()]]), columns=["fitness"], index=[index])
        df = df.append(df_tmp)
df = df.iloc[1:, :]
df = df.sort_values("fitness")
print(np.mean(df.loc[:, 'fitness']))

n_fittest = 2
fidx = [df.index[-i] for i in range(1, n_fittest+1)]

# load parameter set of fittest candidates
##########################################

k = 1000.0
eta = 1000.0
delta = 100.0

dv = 'p'
ivs = ["tau_e", "tau_p", "tau_ampa_r", "tau_ampa_d", "tau_gabaa_r", "tau_gabaa_d", "tau_stn", "eta_e", "eta_p",
       "delta_e", "delta_p", "k_pe", "k_ep", "k_pp", "k_ee"]
param_grid = pd.DataFrame(data=np.zeros((1, len(ivs))), columns=ivs)
for idx in fidx:
    fname = f"{directory}/{fid}_{idx}.h5"
    f = h5py.File(fname, 'r')
    data = {key: f[dv][key][()] for key in ivs[:-1]}
    param_grid_tmp = {
        'tau_e': [data['tau_e']],
        'tau_p': [data['tau_p']],
        'tau_ampa_r': [data['tau_ampa_r']],
        'tau_ampa_d': [data['tau_ampa_d']],
        'tau_gabaa_r': [data['tau_gabaa_r']],
        'tau_gabaa_d': [data['tau_gabaa_d']],
        'tau_stn': [data['tau_stn']],
        'eta_e': [data['eta_e'] * eta],
        'eta_p': [data['eta_p'] * eta],
        'delta_e': [data['delta_e'] * delta],
        'delta_p': [data['delta_e'] * data['delta_p'] * delta],
        'k_pe': [data['k_pe'] * k],
        'k_ep': [data['k_ep'] * k],
        'k_pp': [data['k_ep'] * data['k_pp'] * k],
        'k_ee': [0.1 * data['k_pe'] * k],
    }
    param_grid = param_grid.append(pd.DataFrame.from_dict(param_grid_tmp))
param_grid = param_grid.iloc[1:, :]
param_grid.index = list(range(1, n_fittest+1))

# simulation parameter definitions
##################################

# simulation parameters
dt = 1e-3
dts = 1e-1
T = 4000.0
cutoff = 2000.0

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
    'tau_ampa_d': {'vars': ['gpe_proto_syns_op/tau_ampa_d', 'stn_syns_op/tau_ampa_d'], 'nodes': ['gpe_p', 'stn']},
    'tau_gabaa_d': {'vars': ['gpe_proto_syns_op/tau_gabaa_d', 'stn_syns_op/tau_gabaa_d'], 'nodes': ['gpe_p', 'stn']},
    'tau_ampa_r': {'vars': ['gpe_proto_syns_op/tau_ampa_d', 'stn_syns_op/tau_ampa_d'], 'nodes': ['gpe_p', 'stn']},
    'tau_gabaa_r': {'vars': ['gpe_proto_syns_op/tau_gabaa_d', 'stn_syns_op/tau_gabaa_d'], 'nodes': ['gpe_p', 'stn']},
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
    permute=False,
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

results = results.loc[cutoff:, :]*1e3
results.index = results.index*1e-3
results.plot()
show()

# save results
pickle.dump({'results': results, 'map': result_map}, open(f"results/{fid}_sims.p", "wb"))
