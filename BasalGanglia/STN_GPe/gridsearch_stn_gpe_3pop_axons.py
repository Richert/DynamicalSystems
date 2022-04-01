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
T = 6000.0
cutoff = 1000.0
sim_steps = int(np.round(T/dt))

# parameter sweeps
pa1 = 'k_d'
pa2 = 'k_s'
sweep_params = {pa1: [2.0, 3.0, 4.0, 5.0, 6.0],
                pa2: [0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
                }

# model parameter
k_gp = 9.0
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
        pa1: [],
        pa2: [],
    }

for p1 in sweep_params[pa1]:
    for p2 in sweep_params[pa2]:
        param_grid['k_ee'].append(0.8*k)
        param_grid['k_ae'].append(3.0*k)
        param_grid['k_pe'].append(8.0*k)
        param_grid['k_ep'].append(10.0*k)
        param_grid['k_pp'].append(1.0*k_gp*k_p*k/k_i)
        param_grid['k_ap'].append(1.0*k_gp*k_p*k_i*k)
        param_grid['k_aa'].append(1.0*k_gp*k/(k_p*k_i))
        param_grid['k_pa'].append(1.0*k_gp*k*k_i/k_p)
        param_grid['k_ps'].append(20.0*k)
        param_grid['k_as'].append(20.0*k)
        param_grid['eta_e'].append(4*eta)
        param_grid['eta_p'].append(4*eta)
        param_grid['eta_a'].append(1.0*eta)
        param_grid['eta_s'].append(0.002)
        param_grid['delta_e'].append(30.0)
        param_grid['delta_p'].append(90.0)
        param_grid['delta_a'].append(120.0)
        param_grid['tau_e'].append(13.0)
        param_grid['tau_p'].append(25.0)
        param_grid['tau_a'].append(20.0)
        param_grid[pa1].append(p1)
        param_grid[pa2].append(p2)

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
    pa1: {'vars': ['delay'], 'edges': [('stn', 'gpe_p'), ('gpe_p', 'stn')]},
    pa2: {'vars': ['spread'], 'edges': [('stn', 'gpe_p'), ('gpe_p', 'stn')]}
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
psds = {'freq_stn': [], 'pow_stn': [], 'freq_gpe': [], 'pow_gpe': [], 'fr_stn': [], 'fr_gpe': []}
params = {pa1: [], pa2: []}

for key in result_map.index:

    gpe = results.loc[:, ('r_i', key)]
    p1 = result_map.loc[key, pa1]
    p2 = result_map.loc[key, pa2]

    # calculate PSDs
    # p_stn, f_stn = welch(stn, fmin=2.0, fmax=160.0, n_fft=4096, n_overlap=256)
    p_gpe, f_gpe = welch(gpe, fmin=2.0, fmax=160.0, n_fft=4096, n_overlap=256)

    # store output quantities
    # psds['freq_stn'].append(np.squeeze(f_stn))
    # psds['pow_stn'].append(np.squeeze(p_stn))
    psds['freq_gpe'].append(np.squeeze(f_gpe))
    psds['pow_gpe'].append(np.squeeze(p_gpe))
    # psds['fr_stn'].append(np.squeeze(np.max(stn)-np.min(stn)))
    psds['fr_gpe'].append(np.squeeze(np.max(gpe) - np.min(gpe)))
    params[pa1].append(p1)
    params[pa2].append(p2)

# plot results
pows = np.zeros((len(sweep_params[pa1]), len(sweep_params[pa2])))
freqs = np.zeros_like(pows)
fr_ranges = np.zeros_like(pows)
for i in range(len(params[pa1])):

    row = np.argmin(np.abs(params[pa1][i] - sweep_params[pa1]))
    col = np.argmin(np.abs(params[pa2][i] - sweep_params[pa2]))

    max_idx = np.argmax(psds['pow_gpe'][i])
    pows[row, col] = psds['pow_gpe'][i][max_idx]
    freqs[row, col] = psds['freq_gpe'][i][max_idx]
    fr_ranges[row, col] = psds['fr_gpe'][i]

fig1, ax1 = plt.subplots()
ax = plot_connectivity(freqs, yticklabels=sweep_params[pa1], xticklabels=sweep_params[pa2], ax=ax1)
ax.set_title('f in Hz')
ax.invert_yaxis()
ax.set_xlabel(pa2)
ax.set_ylabel(pa1)

fig2, ax2 = plt.subplots()
ax2 = plot_connectivity(fr_ranges, yticklabels=sweep_params[pa1], xticklabels=sweep_params[pa2], ax=ax2)
ax2.set_title('Range of FRs')
ax2.invert_yaxis()
ax2.set_xlabel(pa2)
ax2.set_ylabel(pa1)

plt.show()

# save results
pickle.dump({'results': results, 'map': result_map, 'psds': psds}, open("results/stn_gpe_3pop_axons_gs.p", "wb"))
