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


# def simulator(x: np.ndarray, x_indices: list, y: np.ndarray, func: Callable, func_args: list,
#               T: float, dt: float, dts: float, cutoff: float, p: float, sigma: float, burst_width: float,
#               burst_sep: float, burst_height: float, width_at_height: float, waveform_length: int,
#               return_dynamics: bool = False):
#
#     # update parameter vector
#     idx_half = int(len(x)/2)
#     for i, j in enumerate(x_indices):
#         func_args[j][0] = x[i]
#         func_args[j][1] = x[idx_half + i]
#
#     # simulate model dynamics
#     fr = integrate(func, tuple(func_args), T + cutoff, dt, dts, cutoff, p) * 1e3
#
#     # get bursting stats
#     res = get_bursting_stats(fr, sigma=sigma, burst_width=burst_width, rel_burst_height=burst_height,
#                              burst_sep=burst_sep, width_at_height=width_at_height, waveform_length=waveform_length)
#
#     # calculate loss
#     y_fit = res["waveform_mean"]
#     y_fit /= np.max(y_fit)
#     loss = sse(y_fit, y)
#
#     if return_dynamics:
#         return loss, res, y_fit
#     return loss

def simulator(x: np.ndarray, x_indices: list, y_target: np.ndarray, func: Callable, func_args: list,
              T: float, dt: float, dts: float, cutoff: float, waveform_length: int, return_dynamics: bool = False):

    # update parameter vector
    for i, j in enumerate(x_indices):
        func_args[j] = x[i]

    # simulate model dynamics
    fr = integrate(func_args[1], func, func_args[2:], T + cutoff, dt, dts, cutoff) * 1e3

    # get waveform
    max_idx_model = np.argmax(fr)
    max_idx_target = np.argmax(y_target)
    start = max_idx_model - max_idx_target
    if start < 0:
        start = 0
    if start + waveform_length > fr.shape[0]:
        start = fr.shape[0] - waveform_length
    y_fit = fr[start:start + waveform_length]
    y_max = np.max(y_fit)
    if y_max > 0:
        y_fit = y_fit / y_max

    # calculate loss
    loss = float(np.sum((y_target - y_fit)**2))

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
n_clusters = 2
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
workers = 80
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
proto_waves = get_cluster_prototypes(clusters.squeeze(), data, reduction_method="random")
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
dts = 1.0
dt = 5e-2
cutoff = 2000.0
T = 10000.0 + cutoff

# exc parameters
params = {
    'tau': 10.0, 'Delta': 2.0, 'eta': -0.5, 'kappa': 0.01, 's': 1.0, 'theta': 20.0, 'tau_a': 500.0, 'J_e': 30.0
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
    "tau": (1.0, 30.0),
    "Delta": (0.1, 20.0),
    "eta": (-10.0, 10.0),
    "kappa": (0.0, 1.0),
    "s": (0.1, 10.0),
    "theta": (0.0, 100.0),
    "tau_a": (100.0, 1000.0),
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
