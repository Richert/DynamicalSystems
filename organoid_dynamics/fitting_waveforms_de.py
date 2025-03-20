import pickle
import numpy as np
from typing import Callable
from custom_functions import *
from pyrates import CircuitTemplate
from scipy.optimize import differential_evolution
from scipy.ndimage import gaussian_filter1d
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import squareform
from numba import njit
import warnings
from time import perf_counter
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=RuntimeWarning)


def integrate(y: np.ndarray, func, args, T, dt, dts, cutoff):

    idx = 0
    steps = int(T / dt)
    cutoff_steps = int(cutoff / dt)
    store_step = int(dts / dt)
    state_rec = []

    # solve ivp for forward Euler method
    for step in range(steps):
        if step > cutoff_steps and step % store_step == 0:
            state_rec.append(y[:2])
            idx += 1
        if not np.isfinite(y[0]):
            state_rec = np.zeros((int((T-cutoff)/dts), y.shape[0]))
            state_rec[int(0.5*state_rec.shape[0]), :] = 1.0
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
    idx_half = int(len(x)/2)
    for i, j in enumerate(x_indices):
        func_args[j][0] = x[i]
        func_args[j][1] = x[idx_half + i]

    # simulate model dynamics
    fr = integrate(func_args[1], func, func_args[2:], T + cutoff, dt, dts, cutoff) * 1e3
    fr = x[-1] * fr[:, 0] + (1 - x[-1]) * fr[:, 1]

    # get waveform
    max_idx_model = np.argmax(fr)
    max_idx_target = np.argmax(y_target)
    start = max_idx_model - max_idx_target
    if start < 0:
        start = 0
    if start + waveform_length > fr.shape[0]:
        start = fr.shape[0] - waveform_length
    y_fit = fr[start:start + waveform_length]
    y_fit = y_fit / np.max(y_fit)

    # calculate loss
    loss = float(np.mean((y_target - y_fit) ** 2))

    if return_dynamics:
        return loss, y_fit
    return loss

# parameter definitions
#######################

# choose device
device = "cpu"

# choose data to fit
dataset = "trujilo_2019"
n_clusters = 9
prototype = 2

# define directories and file to fit
path = "/home/richard-gast/Documents"
save_dir = f"{path}/results/{dataset}"
load_dir = f"{path}/data/{dataset}"

# choose model
model = "ik_eic_sfa"
exc_op = "ik_sfa_op"
inh_op = "ik_sfa_op"

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
maxiter = 200
popsize = 30
mutation = (0.1, 1.2)
recombination = 0.7
polish = False
tolerance = 1e-4

# data loading and processing
#############################

# load data from file
data = pd.read_csv(f"{load_dir}/{dataset}_waveforms.csv", header=[0, 1, 2], index_col=0)
D = np.load(f"{load_dir}/{dataset}_waveform_distances.npy")

# run hierarchical clustering on distance matrix
D_condensed = squareform(D)
Z = linkage(D_condensed, method="ward")
clusters = cut_tree(Z, n_clusters=n_clusters)

# extract target waveform
proto_waves = get_cluster_prototypes(clusters.squeeze(), data, method="random")
y_target = proto_waves[prototype] / np.max(proto_waves[prototype])

plt.plot(y_target)
plt.show()

# model initialization
######################

# simulation parameters
dts = 1.0
dt = 1e-1
cutoff = 1000.0
T = 10000.0 + cutoff

# exc parameters
exc_params = {
    'C': 100.0, 'k': 0.7, 'v_r': -60.0, 'v_t': -40.0, 'Delta': 2.0, 'eta': 70.0, 'kappa': 10.0, 'tau_u': 500.0,
    'g_e': 150.0, 'g_i': 100.0, 'tau_s': 5.0
}

# inh parameters
inh_params = {
    'C': 150.0, 'k': 0.5, 'v_r': -65.0, 'v_t': -45.0, 'Delta': 4.0, 'eta': 0.0, 'kappa': 20.0, 'tau_u': 100.0,
    'g_e': 60.0, 'g_i': 0.0, 'tau_s': 8.0
}

# initialize model template and set fixed parameters
template = CircuitTemplate.from_yaml(f"config/ik_mf/{model}")
template.update_var(node_vars={f"exc/{exc_op}/{key}": val for key, val in exc_params.items()})
template.update_var(node_vars={f"inh/{inh_op}/{key}": val for key, val in inh_params.items()})

# generate run function
inp = np.zeros((int((T + cutoff)/dt),))
func, args, arg_keys, _ = template.get_run_func(f"{model}_vectorfield", step_size=dt, backend="numpy", solver="heun")
func_jit = njit(func)

# free parameter bounds
exc_bounds = {
    "Delta": (0.5, 5.0),
    "k": (0.5, 1.0),
    "kappa": (0.0, 40.0),
    "tau_u": (400.0, 1000.0),
    "g_e": (50.0, 200.0),
    "g_i": (20.0, 200.0),
    "eta": (40.0, 100.0),
    "tau_s": (1.0, 10.0),
}
inh_bounds = {
    "Delta": (2.0, 8.0),
    "k": (0.2, 1.0),
    "kappa": (0.0, 80.0),
    "tau_u": (100.0, 500.0),
    "g_e": (40.0, 150.0),
    "g_i": (0.0, 100.0),
    "eta": (-50.0, 50.0),
    "tau_s": (5.0, 20.0),
}

# find argument positions of free parameters
param_indices = []
for key in list(exc_bounds.keys()):
    idx = arg_keys.index(f"exc/{exc_op}/{key}")
    param_indices.append(idx)

# define final arguments of loss/simulation function
func_args = (param_indices, y_target, func_jit, list(args), T, dt, dts, cutoff, waveform_length)

# fitting procedure
###################

# combine parameter bounds
p_e = 0.9
x0 = []
bounds = []
param_keys = []
for b, group in zip([exc_bounds, inh_bounds], ["exc", "inh"]):
    for key, (low, high) in b.items():
        x0.append(0.5*(low + high))
        bounds.append((low, high))
        param_keys.append(f"{group}/{key}")
x0.append(p_e)
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
                                     recombination=recombination, polish=polish, atol=tolerance)
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
loss, _, y_fit = simulator(results.x, *func_args, return_dynamics=True)

# save results
pickle.dump({
    "target_waveform": y_target, "fitted_waveform": y_fit, "fitted_parameters": fitted_parameters},
    open(f"{save_dir}/{dataset}_prototype_{prototype}_fit.pkl", "wb"))
