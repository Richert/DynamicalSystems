import pickle
import numpy as np
from scipy.io import loadmat
from typing import Callable
import os
from custom_functions import *
from pyrates import CircuitTemplate
from scipy.optimize import differential_evolution
from scipy.ndimage import gaussian_filter1d
from numba import njit
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def integrate(func, func_args, T, dt, dts, cutoff):

    t, y = float(func_args[0]), func_args[1]
    args = func_args[2:]
    fs = int(dts/dt)
    y_col = []
    step = 0
    while t < T:
        y += dt * func(step, y, *args)
        t += dt
        step += 1
        if t >= cutoff and step % fs == 0:
            y_col.append(y[0])
            if not np.isfinite(y):
                y_col[-1] = np.inf
                break
    return np.asarray(y_col)


def simulator(x: np.ndarray, x_indices: list, y: np.ndarray, func: Callable, func_args: list, inp_idx: int,
              T: float, dt: float, dts: float, cutoff: float, nperseg: int, fmax: float, sigma: float,
              burst_width: float, burst_height: float, return_dynamics: bool = False):

    # define extrinsic input
    s_ext, noise_lvl, noise_sigma = x[-3:]
    inp = np.zeros((int((T + cutoff)/dt),))
    noise = noise_lvl * np.random.randn(inp.shape[0])
    noise = gaussian_filter1d(noise, sigma=noise_sigma)
    inp += s_ext + noise

    # update parameter vector
    for idx, val in zip(x_indices, x):
        func_args[idx] = val
    func_args[inp_idx] = inp

    # simulate model dynamics
    fr = integrate(func, tuple(func_args), T + cutoff, dt, dts, cutoff) * 1e3

    # create summary statistics
    if np.isfinite(fr[-1]):
        freqs, fitted_psd = get_psd(fr, fs=1e3/dts, nperseg=nperseg, fmax=fmax, detrend=True)
        burst_stats = get_bursting_stats(fr, sigma=sigma, burst_width=burst_width, rel_burst_height=burst_height,
                                         width_at_height=0.9)
        bursting = np.asarray([burst_stats["ibi_mean"], burst_stats["ibi_std"]])

        sum_stats = np.concatenate([fitted_psd, bursting], axis=0)
    else:
        sum_stats = np.zeros_like(y) + np.inf

    # calculate loss
    loss = mse(sum_stats, y)

    if return_dynamics:
        return loss, sum_stats, fr
    return loss


# data set specifics
####################

# parameter definitions
#######################

device = "cpu"
save_dir = "/home/richard/results"

# choose data set
dataset_name = "trujilo_2019"
path = f"/home/richard/data/{dataset_name}"
file = "161001"

# choose model
model = "ik_sfa"
input_var = "I_ext"

# dataset parameters
well = 4
well_offset = 4
dt = 1e-1
cutoff = 500.0

# data processing parameters
tau = 20.0
nperseg = 50000
fmax = 100.0
detrend = True
sigma = 100.0
burst_width = 1000.0
burst_height = 0.5

# fitting parameters
strategy = "best1exp"
workers = 80
maxiter = 1000
popsize = 50
mutation = (0.5, 1.5)
recombination = 0.6
polish = True
tolerance = 1e-3

# data loading and processing
#############################

# prepare age calculation
year_0 = 16
month_0 = 7
day_0 = 1

# load data from file
data = loadmat(f"{path}/LFP_Sp_{file}.mat", squeeze_me=False)

# extract data
lfp = data["LFP"]
spike_times = data["spikes"]
time = np.squeeze(data["t_s"])
time_ds = np.round(np.squeeze(data["t_ds"]), decimals=4) # type: np.ndarray

# calculate organoid age
date = file.split(".")[0].split("_")[-1]
year, month, day = int(date[:2]), int(date[2:4]), int(date[4:])
month += int((year-year_0)*12)
day += int((month-month_0)*30)
age = day - day_0

# calculate firing rate
T = time_ds[-1] * 1e3
dts = float(time_ds[1] - time_ds[0]) * 1e3
spikes = extract_spikes(time, time_ds, spike_times[well + well_offset - 1])
spikes_smoothed = convolve_exp(spikes, tau=tau, dt=dts, normalize=False)
target_fr = np.mean(spikes_smoothed, axis=0) / tau

# calculate psd
freqs, target_psd = get_psd(target_fr, fs=1e3/dts, nperseg=nperseg, fmax=fmax, detrend=detrend)

# calculate bursting stats
target_bursts = get_bursting_stats(target_fr, sigma=sigma, burst_width=burst_width, rel_burst_height=burst_height,
                                   width_at_height=0.9)
target_bursts = np.asarray([target_bursts["ibi_mean"], target_bursts["ibi_std"]])

# combine to target summary stats
y_target = np.concatenate([target_psd, target_bursts], axis=0)

# model initialization
######################

# fixed model parameters
model_params = {
    "v_r": -60.0,
    "v_t": -40.0,
    "eta": 0.0,
    "E_e": 0.0,
    "tau_s": 8.0,
    "I_ext": 0.0,
}

# free parameter bounds
bounds = {
    "C": (80.0, 300.0),
    "k": (0.5, 1.5),
    "Delta": (0.1, 2.0),
    "kappa": (0.5, 2.0),
    "tau_u": (400.0, 1000.0),
    "g_e": (5.0, 30.0),
    "s_ext": (100.0, 300.0),
    "noise_lvl": (5.0, 80.0),
    "sigma": (30.0, 100.0)
}

# initialize model template and set fixed parameters
template = CircuitTemplate.from_yaml(f"config/ik_mf/{model}")
template.update_var(node_vars={f"p/{model}_op/{key}": val for key, val in model_params.items()})

# generate run function
inp = np.zeros((int((T + cutoff)/dt),))
func, args, arg_keys, _ = template.get_run_func(f"{model}_vectorfield", step_size=dt,
                                                inputs={f'p/{model}_op/{input_var}': inp},
                                                backend="numpy", solver="heun")
func_jit = njit(func)
func_jit(*args)

# find argument positions of free parameters
param_indices = []
for key in list(bounds.keys())[:-3]:
    idx = arg_keys.index(f"p/{model}_op/{key}")
    param_indices.append(idx)
input_idx = arg_keys.index(f"{input_var}_input_node/{input_var}_input_op/{input_var}_input")

# fitting procedure
###################

# fitting procedure
func_args = (param_indices, y_target, func, list(args), input_idx, T, dt, dts, cutoff, nperseg, fmax, sigma,
             burst_width, burst_height)
while True:
    results = differential_evolution(simulator, tuple(bounds.values()), args=func_args, strategy=strategy,
                                     workers=workers, disp=True, maxiter=maxiter, popsize=popsize, mutation=mutation,
                                     recombination=recombination, polish=polish, atol=tolerance)
    if np.isnan(results.fun):
        print("Re-initializing. Reason: loss(best candidate) = NaN.")
    else:
        break
print("And the winner is ... ")
fitted_parameters = {}
for key, val in zip(bounds.keys(), results.x):
    print(f"{key} = {val}")
    fitted_parameters[key] = val

# generate dynamics of winner
loss, y_fit, fr_fit = simulator(results.x, *func_args, return_dynamics=True)

# save results
pickle.dump({
    "age": age, "organoid": well, "freqs": freqs, "time": time_ds,
    "target_psd": target_psd, "target_bursts": target_bursts, "target_fr": target_fr,
    "fitted_psd": y_fit[:-3], "fitted_bursts": y_fit[-3:], "fitted_fr": fr_fit,
    "fitted_parameters": fitted_parameters},
    open(f"{save_dir}/{dataset_name}/{file}_de_fit.pkl", "wb"))
