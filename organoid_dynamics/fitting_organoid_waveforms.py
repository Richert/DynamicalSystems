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


def integrate(func, func_args, T, dt, dts, cutoff, p):

    t0, y = int(func_args[0]), func_args[1]
    args = func_args[2:]
    idx = 0
    steps = int(np.round(T / dt))
    cutoff_steps = int(np.round(cutoff / dt))
    store_steps = int(np.round((T-cutoff) / dts))
    store_step = int(np.round(dts / dt))
    state_rec = np.zeros((store_steps,), dtype=y.dtype)

    # solve ivp for forward Euler method
    for step in range(t0, steps + t0):
        r_e, r_i = y[0], y[6]
        if step > cutoff_steps and step % store_step == t0:
            state_rec[idx] = p*r_e + (1-p)*r_i
            idx += 1
        if not np.isfinite(r_e):
            break
        rhs = func(step, y, *args)
        y_0 = y + dt * rhs
        y += (rhs + func(step, y_0, *args)) * dt/2

    return state_rec


def simulator(x: np.ndarray, x_indices: list, y: np.ndarray, func: Callable, func_args: list, inp_idx: int,
              T: float, dt: float, dts: float, cutoff: float, p: float, sigma: float, burst_width: float,
              burst_sep: float, burst_height: float, width_at_height: float, waveform_length: int,
              return_dynamics: bool = False):

    # define extrinsic input
    noise_lvl, noise_sigma = x[-2:]
    inp = np.zeros((int((T + cutoff)/dt) + 1,))
    noise = noise_lvl * np.random.randn(inp.shape[0])
    inp += gaussian_filter1d(noise, sigma=noise_sigma)

    # update parameter vector
    for idx, val in zip(x_indices, x):
        func_args[idx] = val
    func_args[inp_idx] = inp

    # simulate model dynamics
    fr = integrate(func, tuple(func_args), T + cutoff, dt, dts, cutoff, p) * 1e3

    # get bursting stats
    res = get_bursting_stats(fr, sigma=sigma, burst_width=burst_width, rel_burst_height=burst_height,
                             burst_sep=burst_sep, width_at_height=width_at_height, waveform_length=waveform_length)

    # calculate loss
    y_fit = res["waveform_mean"]
    loss = mse(y_fit / np.max(y_fit), y / np.max(y))

    if return_dynamics:
        return loss, res, y_fit
    return loss

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
model = "eic_ik"
exc_op = "ik_sfa_op"
inh_op = "ik_op"
input_var = "I_ext"

# data processing parameters
tau = 20.0
sigma = 20.0
burst_width = 100.0
burst_sep = 1000.0
burst_height = 0.5
burst_relheight = 0.9
waveform_length = 3000

# fitting parameters
strategy = "best1exp"
workers = 10
maxiter = 1000
popsize = 10
mutation = (0.5, 1.5)
recombination = 0.6
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
clusters = cut_tree(Z, n_clusters=6)

# extract target waveform
prototype = 0
proto_waves = get_cluster_prototypes(clusters.squeeze(), data, method="random")
y_target = proto_waves[prototype]

plt.plot(y_target)
plt.show()

# model initialization
######################

# simulation parameters
dts = 1.0
dt = 1e-1
cutoff = 1000.0
T = 5000.0 + cutoff
p_e = 0.8 # fraction of excitatory neurons

# exc parameters
exc_params = {
    'C': 100.0, 'k': 0.7, 'v_r': -60.0, 'v_t': -40.0, 'Delta': 1.0, 'eta': 70.0, 'kappa': 1000.0, 'tau_u': 1000.0,
    'g_e': 60.0, 'g_i': 20.0, 'tau_s': 6.0
}

# inh parameters
inh_params = {
    'C': 100.0, 'k': 0.7, 'v_r': -60.0, 'v_t': -40.0, 'Delta': 1.0, 'eta': 0.0, 'g_e': 40.0, 'g_i': 0.0, 'tau_s': 20.0
}

# initialize model template and set fixed parameters
template = CircuitTemplate.from_yaml(f"config/ik_mf/{model}")
template.update_var(node_vars={f"exc/{exc_op}/{key}": val for key, val in exc_params.items()})
template.update_var(node_vars={f"inh/{inh_op}/{key}": val for key, val in inh_params.items()})

# generate run function
inp = np.zeros((int((T + cutoff)/dt),))
func, args, arg_keys, _ = template.get_run_func(f"{model}_vectorfield", step_size=dt,
                                                inputs={f'exc/{exc_op}/{input_var}': inp},
                                                backend="numpy", solver="heun")
func_jit = njit(func)

# free parameter bounds
exc_bounds = {
    "Delta": (0.5, 5.0),
    "k": (0.5, 1.5),
    "kappa": (500.0, 3000.0),
    "tau_u": (500.0, 5000.0),
    "g_e": (30.0, 120.0),
    "g_i": (20.0, 120.0),
    "eta": (40.0, 100.0),
    "tau_s": (5.0, 20.0),
}
inh_bounds = {
    "Delta": (0.5, 5.0),
    "k": (0.5, 1.5),
    "g_e": (20.0, 120.0),
    "g_i": (20.0, 120.0),
    "eta": (0.0, 100.0),
    "tau_s": (5.0, 50.0),
}
noise_bounds = {
    "noise_lvl": (10.0, 100.0),
    "sigma": (50.0, 400.0)
}

# find argument positions of free parameters
param_indices = []
for key in list(exc_bounds.keys()):
    idx = arg_keys.index(f"exc/{exc_op}/{key}")
    param_indices.append(idx)
for key in list(inh_bounds.keys()):
    idx = arg_keys.index(f"inh/{inh_op}/{key}")
    param_indices.append(idx)
input_idx = arg_keys.index(f"{input_var}_input_node/{input_var}_input_op/{input_var}_input")

# define final arguments of loss/simulation function
func_args = (param_indices, y_target, func_jit, list(args), input_idx, T, dt, dts, cutoff, p_e, sigma, burst_width,
             burst_sep, burst_height, burst_relheight, waveform_length)

# fitting procedure
###################

# combine parameter bounds
x0 = []
bounds = []
param_keys = []
for b, group in zip([exc_bounds, inh_bounds, noise_bounds], ["exc", "inh", "noise"]):
    for key, (low, high) in b.items():
        x0.append(0.5*(low + high))
        bounds.append((low, high))
        param_keys.append(f"{group}/{key}")
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
