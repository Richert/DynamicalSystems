import pickle
import numpy as np
from typing import Callable
from custom_functions import *
from pyrates import CircuitTemplate
from scipy.optimize import differential_evolution
from scipy.ndimage import gaussian_filter1d
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import squareform
from sklearn.preprocessing import MinMaxScaler
from tslearn.metrics import cdist_dtw
from numba import njit
import warnings
from time import perf_counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
warnings.filterwarnings("ignore", category=RuntimeWarning)

def callback(intermediate_result):
    if intermediate_result.fun < tolerance:
        return True
    else:
        return False


def integrate(y: np.ndarray, func, args, T, dt, dts, cutoff):

    idx = 0
    steps = int(T / dt)
    cutoff_steps = int(cutoff / dt)
    store_step = int(dts / dt)
    store_steps = int((T - cutoff) / dts)
    state_rec = []

    # solve ivp for forward Euler method
    for step in range(steps):
        if step > cutoff_steps and step % store_step == 0:
            state_rec.append(y[0])
            idx += 1
        if not np.isfinite(y[0]):
            n_zeros = store_steps - len(state_rec)
            y0 = 0.0
            state_rec.extend([y0] * n_zeros)
            break
        rhs = func(step, y, *args)
        y_0 = y + dt * rhs
        y = y + (rhs + func(step, y_0, *args)) * dt/2

    return np.asarray(state_rec)


def simulator(x: np.ndarray, x_indices: list, y_target: np.ndarray, func: Callable, func_args: list,
              T: float, dt: float, dts: float, cutoff: float, waveform_length: int, return_dynamics: bool = False):

    # update parameter vector
    for i, j in enumerate(x_indices):
        func_args[j] = x[i]

    # simulate model dynamics
    fr = integrate(func_args[1], func, func_args[2:], T + cutoff, dt, dts, cutoff)
    fr_max = np.max(fr)
    if fr_max > 0:
        fr = fr / fr_max

    # get waveform
    idx = 0
    correlations = []
    while idx + waveform_length < len(fr):
        correlations.append(np.dot(y_target, fr[idx:idx+waveform_length]))
        idx += 1
    max_idx, max_corr = np.argmax(correlations), np.max(correlations)
    y_fit = fr[max_idx:max_idx + waveform_length]
    loss = 1 - max_corr / np.dot(y_target, y_target)

    if return_dynamics:
        return loss, y_fit
    return loss

# parameter definitions
#######################

# choose device
device = "cpu"
n_jobs = 15

# choose data to fit
dataset = "trujilo_2019"
n_clusters = 4
prototype = 1

# define directories and file to fit
path = "/home/richard-gast/Documents"
save_dir = f"{path}/results/{dataset}"
load_dir = f"{path}/data/{dataset}"

# choose model
model = "qif_ca"
op = "qif_ca_op"

# data processing parameters
sigma = 20.0
burst_width = 100.0
burst_sep = 1000.0
burst_height = 0.5
burst_relheight = 0.9
waveform_length = 3000

# fitting parameters
strategy = "best1exp"
workers = 15
maxiter = 300
popsize = 30
mutation = (0.1, 1.2)
recombination = 0.7
polish = False
tolerance = 1e-1

# data loading and processing
#############################

# load data from file
data = pd.read_csv(f"{load_dir}/{dataset}_waveforms.csv", header=[0, 1, 2], index_col=0)

# reduce data
age = 82
organoid = None
normalize = True
data = reduce_df(data, age=age, organoid=organoid)
data_norm = data.values.T
if normalize:
    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(data_norm)
D = cdist_dtw(data_norm[:, :], n_jobs=n_jobs)

# run hierarchical clustering on distance matrix
D_condensed = squareform(D)
Z = linkage(D_condensed, method="ward")
clusters = cut_tree(Z, n_clusters=n_clusters)

# extract target waveform
proto_waves = get_cluster_prototypes(clusters.squeeze(), data, reduction_method="mean")
y_target = proto_waves[prototype] / np.max(proto_waves[prototype])

# plot prototypical waveforms
fig, ax = plt.subplots(figsize=(12, 5))
for sample, wave in proto_waves.items():
    ax.plot(wave / np.max(wave), label=sample)
ax.set_ylabel("firing rate")
ax.set_xlabel("time (ms)")
ax.legend()
ax.set_title(f"Normalized cluster waveforms")
plt.tight_layout()

# plot clustering results
ax = sb.clustermap(D, row_linkage=Z, figsize=(12, 9))
plt.title(f"Distance matrix and dendrogram")
plt.tight_layout()
plt.show()

# model initialization
######################

# simulation parameters
dts = 1e-1
dt = 1e-3
cutoff = 200.0
T = 800.0 + cutoff

# exc parameters
params = {
    'tau': 10.0, 'Delta': 2.0, 'eta': -0.5, 'kappa': 0.01, 'tau_a': 500.0, 'J_e': 30.0
}

# initialize model template and set fixed parameters
template = CircuitTemplate.from_yaml(f"config/ik_mf/{model}")
template.update_var(node_vars={f"p/{op}/{key}": val for key, val in params.items()})

# generate run function
inp = np.zeros((int((T + cutoff)/dt),))
func, args, arg_keys, _ = template.get_run_func(f"{model}_vectorfield", step_size=dt, backend="numpy", solver="heun",
                                                float_precision="float64", vectorize=False)
func_jit = njit(func)

# free parameter bounds
param_bounds = {
    "tau": (1.0, 5.0),
    "Delta": (0.1, 5.0),
    "eta": (-10.0, 10.0),
    "kappa": (0.0, 1.0),
    "tau_a": (10.0, 100.0),
    "J_e": (1.0, 100.0),
}

# find argument positions of free parameters
param_indices = []
for key in list(param_bounds.keys()):
    idx = arg_keys.index(f"p/{op}/{key}")
    param_indices.append(idx)

# define final arguments of loss/simulation function
func_args = (param_indices, y_target, func_jit, list(args), T, dt, dts, cutoff, waveform_length)

# fitting procedure
###################

# combine parameter bounds
x0 = []
bounds = []
param_keys = []
for key, (low, high) in param_bounds.items():
    x0.append(0.5*(low + high))
    bounds.append((low, high))
    param_keys.append(f"{key}")
x0 = np.asarray(x0)

# test run
print(f"Starting a test run of the mean-field model for {np.round(T, decimals=0)} ms simulation time, using a "
      f"simulation step-size of {dt} ms.")
t0 = perf_counter()
simulator(x0, *func_args, return_dynamics=False)
t1 = perf_counter()
print(f"Finished test run after {t1-t0} s.")

# fitting procedure
print(f"Starting to fit the mean-field model to {np.round(T, decimals=0)} ms of spike recordings ...")
while True:
    results = differential_evolution(simulator, tuple(bounds), args=func_args, strategy=strategy,
                                     workers=workers, disp=True, maxiter=maxiter, popsize=popsize, mutation=mutation,
                                     recombination=recombination, polish=polish, callback=callback)
    if np.isnan(results.fun):
        print("Re-initializing. Reason: loss(best candidate) = NaN.")
    else:
        break
print(f"Finished fitting procedure. The winner (loss = {results.fun}) is ... ")
fitted_parameters = {}
for key, val in zip(param_keys, results.x):
    print(f"{key} = {val}")
    fitted_parameters[key] = val

# generate dynamics of winner
loss, y_fit = simulator(results.x, *func_args, return_dynamics=True)

# save results
pickle.dump({
    "target_waveform": y_target, "fitted_waveform": y_fit, "fitted_parameters": fitted_parameters},
    open(f"{save_dir}/{dataset}_prototype_{prototype}_age_{age}_fit.pkl", "wb"))
