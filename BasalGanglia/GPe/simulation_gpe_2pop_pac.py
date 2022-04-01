import matplotlib.pyplot as plt
from pyrates.utility.grid_search import grid_search
from pyrates.utility.data_analysis import welch
import pickle

# parameter definitions
#######################

condition = 'bs'

# simulation parameters
dt = 1e-3
dts = 1e-1
T = 12050.0
cutoff = 2.0

# stimulation parametedx:dt::x**2-I
# rs
# if condition == 'bs':
#     stim_periods = [59.0, 63.0, 67.0, 72.0]
#     stim_amps = [1.0, 0.1, 1.0, 1.0]
# elif condition == 'lc':
#     stim_periods = [56.0, 56.0, 72.0, 74.0]
#     stim_amps = [1.1, 0.2, 0.8, 1.3]
# else:
#     stim_periods = [55.0, 55.0, 65.0, 65.0]
#     stim_amps = [0.1, 1.0, 0.5, 1.5]
stim_periods = [56.0, 56.0, 66.0, 69.0]
stim_amps = [0.3, 1.1, 1.1, 1.1]
n = len(stim_periods)

# model parameters
k_gp = 1.0
k = 10.0
param_grid = {
        'k_ae': [k*1.5]*n,
        'k_pe': [k*5.0]*n,
        'k_pp': [1.5*k*k_gp]*n,
        'k_ap': [2.0*k*k_gp]*n,
        'k_aa': [0.1*k*k_gp]*n,
        'k_pa': [0.5*k*k_gp]*n,
        'k_ps': [k*10.0]*n,
        'k_as': [k*1.0]*n,
        'eta_e': [0.02]*n,
        'eta_p': [12.0]*n,
        'eta_a': [26.0]*n,
        'eta_s': [0.002]*n,
        'delta_p': [9.0]*n,
        'delta_a': [3.0]*n,
        'tau_p': [18]*n,
        'tau_a': [32]*n,
        'omega': stim_periods,
        'a2': stim_amps,
        'a1': [0.0]*n
    }

if condition == 'lc':
    param_grid['k_pp'] = [5.0*k]*n
    param_grid['eta_p'] = [40.0]*n
elif condition == 'bs':
    param_grid['k_pa'] = [5.0*k] * n
    param_grid['eta_p'] = [24.0] * n

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
    'omega': {'vars': ['sl_op/t_off'], 'nodes': ['driver']},
    'a1': {'vars': ['weight'], 'edges': [('driver', 'gpe_p', 0)]},
    'a2': {'vars': ['weight'], 'edges': [('driver', 'gpe_p', 1)]}
}

# ctx *= param_grid['delta_p']
# plt.plot(ctx)
# plt.show()

# simulations
#############

results, result_map = grid_search(
    circuit_template="/home/rgast/PycharmProjects/BrainNetworks/BasalGanglia/config/stn_gpe/gpe_2pop_driver",
    param_grid=param_grid,
    param_map=param_map,
    simulation_time=T,
    step_size=dt,
    permute_grid=False,
    sampling_step_size=dts,
    inputs={
        #'driver/sl_op/alpha': inp
    },
    outputs={
        'r_i': 'gpe_p/gpe_proto_syns_op/R_i',
        'r_a': 'gpe_a/gpe_arky_syns_op/R_a'
    },
    init_kwargs={
        'backend': 'numpy', 'solver': 'scipy', 'step_size': dt},
    method='RK45'
)

psds = []
rates = []
stims = []
for key in result_map.index:

    # extract rates
    proto = results.loc[cutoff:, ('r_i', key)]
    arky = results.loc[cutoff:, ('r_a', key)]

    # calculate PSDs
    pows, freqs = welch(proto, fmin=1.0, fmax=100.0, tmin=0.0, n_fft=1024, n_overlap=512)

    # store results
    rates.append({'proto': proto, 'arky': arky})
    psds.append({'freq': freqs, 'pow': pows})
    stims.append({'alpha': result_map.at[key, 'a2'], 'omega': result_map.at[key, 'omega']})

# save results
pickle.dump({'rates': rates, 'psds': psds, 'params': stims}, open(f"results/gpe_2pop_pac_{condition}.p", "wb"))

results.plot()
plt.show()
