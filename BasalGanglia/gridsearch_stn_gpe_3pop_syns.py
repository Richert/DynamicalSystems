from pyrates.utility.grid_search import grid_search
from pyrates.utility.data_analysis import welch
from pyrates.utility.visualization import plot_connectivity
import numpy as np
import pickle
import matplotlib.pyplot as plt

# config
########

# parameters
dt = 1e-3
dts = 1e-1
T = 3000.0
cutoff = 1000.0
sim_steps = int(np.round(T/dt))

# parameter sweeps
sweep_params = {'tau_stn': [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0],
                'tau_gabaa': [4.0, 6.0, 8.0, 10.0, 12.0, 14.0]}

# model parameter
k_gp = 10.6
k_p = 2.0
k_i = 1.5
k = 100.0
eta = 100.0
param_grid = {
        'k_ee': [0.8*k],
        'k_ae': [3.0*k],
        'k_pe': [8.0*k],
        'k_ep': [10.0*k],
        'k_pp': [1.0*k_gp*k_p*k/k_i],
        'k_ap': [1.0*k_gp*k_p*k_i*k],
        'k_aa': [1.0*k_gp*k/(k_p*k_i)],
        'k_pa': [0.0*k_gp*k_i*k/k_p],
        'k_ps': [20.0*k],
        'k_as': [20.0*k],
        'eta_e': [4.0*eta],
        'eta_p': [3.145*eta],
        'eta_a': [1.0*eta],
        'eta_s': [0.002],
        'delta_e': [30.0],
        'delta_p': [90.0],
        'delta_a': [120.0],
        'tau_e': [13.0],
        'tau_p': [25.0],
        'tau_a': [20.0],
        'tau_gabaa': sweep_params['tau_gabaa'],
        'tau_stn': sweep_params['tau_stn']
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
    #'tau_ampa': {'vars': ['gpe_proto_syns_op/tau_ampa_d', 'stn_syns_op/tau_ampa_d'], 'nodes': ['gpe_p', 'stn']},
    'tau_gabaa': {'vars': ['gpe_proto_syns_op/tau_gabaa_d', 'stn_syns_op/tau_gabaa_d'], 'nodes': ['gpe_p', 'stn']},
    'tau_stn': {'vars': ['stn_syns_op/tau_gabaa'], 'nodes': ['stn']}
}

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
        'r_e': 'stn/stn_syns_op/R_e',
        'r_i': 'gpe_p/gpe_proto_syns_op/R_i',
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
params = {'tau_stn': [], 'tau_gabaa': []}

for key in result_map.index:

    stn = results.loc[:, ('r_e', key)]
    gpe = results.loc[:, ('r_i', key)]
    tau_a = result_map.loc[key, 'tau_stn']
    tau_g = result_map.loc[key, 'tau_gabaa']

    # calculate PSDs
    p_stn, f_stn = welch(stn, fmin=2.0, fmax=160.0, n_fft=4096, n_overlap=256)
    p_gpe, f_gpe = welch(gpe, fmin=2.0, fmax=160.0, n_fft=4096, n_overlap=256)

    # store output quantities
    psds['freq_stn'].append(np.squeeze(f_stn))
    psds['pow_stn'].append(np.squeeze(p_stn))
    psds['freq_gpe'].append(np.squeeze(f_gpe))
    psds['pow_gpe'].append(np.squeeze(p_gpe))
    params['tau_stn'].append(tau_a)
    params['tau_gabaa'].append(tau_g)

# plot results
pows = np.zeros((len(sweep_params['tau_stn']), len(sweep_params['tau_gabaa'])))
freqs = np.zeros_like(pows)
for i in range(len(params['tau_stn'])):

    row = np.argmin(np.abs(params['tau_stn'][i] - sweep_params['tau_stn']))
    col = np.argmin(np.abs(params['tau_gabaa'][i] - sweep_params['tau_gabaa']))

    max_idx = np.argmax(psds['pow_gpe'][i])
    pows[row, col] = psds['pow_gpe'][i][max_idx]
    freqs[row, col] = psds['freq_gpe'][i][max_idx]

fig1, ax1 = plt.subplots()
ax = plot_connectivity(freqs, yticklabels=sweep_params['tau_stn'], xticklabels=sweep_params['tau_gabaa'], ax=ax1)
ax.set_title('f in Hz')
ax.invert_yaxis()
ax.set_xlabel('tau_gabaa')
ax.set_ylabel('tau_stn')

fig2, ax2 = plt.subplots()
ax2 = plot_connectivity(pows, yticklabels=sweep_params['tau_stn'], xticklabels=sweep_params['tau_gabaa'], ax=ax2)
ax2.set_title('PSD')
ax2.invert_yaxis()
ax2.set_xlabel('tau_gabaa')
ax2.set_ylabel('tau_stn')

plt.show()

# save results
pickle.dump({'results': results, 'map': result_map, 'psds': psds}, open("results/stn_gpe_3pop_syns_gs.p", "wb"))
