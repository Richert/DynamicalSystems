import pickle
import numpy as np
from custom_functions import *
from pyrates import CircuitTemplate
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import squareform
from numba import njit
import warnings
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=RuntimeWarning)

# parameter definitions
#######################

# choose device
device = "cpu"

# choose data set
dataset = "trujilo_2019"

# define directories and file to fit
path = "/home/richard-gast/Documents"
save_dir = f"{path}/results/{dataset}"
load_dir = f"{path}/data/{dataset}"

# choose model
model = "eic_stp"
exc_op = "ik_stp_op"
inh_op = "ik_op"

# data processing parameters
sigma = 10.0
burst_width = 100.0
burst_sep = 1000.0
burst_height = 0.5
burst_relheight = 0.9
waveform_length = 3000

# fitting parameters
strategy = "best1exp"
workers = 15
maxiter = 200
popsize = 20
mutation = (0.5, 1.2)
recombination = 0.7
polish = True
tolerance = 1e-3

# data loading and processing
#############################

# load data from file
data = pd.read_csv(f"{load_dir}/{dataset}_waveforms.csv", header=[0, 1, 2], index_col=0)
D = np.load(f"{load_dir}/{dataset}_waveform_distances.npy")

# run hierarchical clustering on distance matrix
D_condensed = squareform(D)
Z = linkage(D_condensed, method="ward")
clusters = cut_tree(Z, n_clusters=9)

# extract target waveform
prototype = 1
proto_waves = get_cluster_prototypes(clusters.squeeze(), data, reduction_method="random")
y_target = proto_waves[prototype] / np.max(proto_waves[prototype])

# plt.plot(y_target)
# plt.show()

# model initialization
######################

# simulation parameters
dts = 1.0
dt = 1e-1
cutoff = 0.0
T = 12000.0 + cutoff
p_e = 0.8 # fraction of excitatory neurons

# exc parameters
exc_params = {
    'C': 100.0, 'k': 0.7, 'v_r': -60.0, 'v_t': -40.0, 'Delta': 4.0, 'eta': 70.0, 'kappa': 1.0, 'tau_u': 200.0,
    'g_e': 80.0, 'g_i': 80.0, 'tau_s': 6.0, 'alpha': 0.4, 'tau_d': 800.0, 'X0': 0.3, 'tau_f': 100.0
}

# inh parameters
inh_params = {
    'C': 100.0, 'k': 0.7, 'v_r': -60.0, 'v_t': -40.0, 'Delta': 8.0, 'eta': 50.0, 'g_e': 80.0, 'g_i': 80.0, 'tau_s': 10.0
}

# initialize model template and set fixed parameters
template = CircuitTemplate.from_yaml(f"config/ik_mf/{model}")
template.update_var(node_vars={f"exc/{exc_op}/{key}": val for key, val in exc_params.items()})
template.update_var(node_vars={f"inh/{inh_op}/{key}": val for key, val in inh_params.items()})

# waveform simulation
#####################

# run simulation
res_mf = template.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='heun',
                      outputs={'r_e': f'exc/{exc_op}/r', 'r_i': f'inh/{inh_op}/r',
                               'x_e': f'exc/{exc_op}/x_e', 'x_i': f'exc/{exc_op}/x_i'},
                      decorator=njit)
res_mf["r_e"] *= 1e3
res_mf["r_i"] *= 1e3
fig, axes = plt.subplots(nrows=4, figsize=(12, 8))
for idx, key in enumerate(["r_e", "r_i", "x_e", "x_i"]):
    ax = axes[idx]
    ax.plot(res_mf[key])
    ax.set_ylabel(key)
    ax.set_xlabel("time (ms)")
plt.tight_layout()
plt.show()

# get bursting stats
fr = p_e*res_mf["r_e"].values + (1-p_e)*res_mf["r_i"].values
res = get_bursting_stats(fr, sigma=sigma, burst_width=burst_width, rel_burst_height=burst_height,
                         burst_sep=burst_sep, width_at_height=burst_relheight, waveform_length=waveform_length,
                         all_waveforms=True)

y_model = res["waveforms"][0]

# plotting
##########

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(y_target / np.max(y_target), label="target")
ax.plot(y_model / np.max(y_model), label="model")
ax.set_xlabel("time (ms)")
ax.set_ylabel("norm. FR")
ax.set_title(f"Waveform of cluster {prototype}")
ax.legend()
plt.tight_layout()
plt.show()
