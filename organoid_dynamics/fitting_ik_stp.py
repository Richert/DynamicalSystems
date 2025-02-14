import pickle

import numpy as np
from scipy.io import loadmat
from typing import Callable
import os
from custom_functions import *
from pyrates import CircuitTemplate
from scipy.optimize import differential_evolution
from scipy.integrate import solve_ivp
from numba import njit
import matplotlib.pyplot as plt
from time import perf_counter


def callback(intermediate_result=None) -> bool:
    if np.isfinite(intermediate_result.fun):
        return False
    else:
        return True


#@njit
def integrate(func, func_args, T, dt, dts, cutoff):

    t, y = float(func_args[0]), func_args[1]
    args = func_args[2:]
    fs = int(dts/dt)
    y_col = []
    step = 0
    while t <= T:
        y += dt * func(step, y, *args)
        t += dt
        step += 1
        if t >= cutoff and step % fs == 0:
            y_col.append(y[0])
            if not np.isfinite(y[0]):
                break
    return np.asarray(y_col)


def loss_func(x: np.ndarray, x_indices: list, y_psd: np.ndarray, y_bursts: np.ndarray, func: Callable, func_args: list,
              inp_idx: int, time_scale: float, T: float, dt: float, dts: float, cutoff: float, nperseg: int, fmax: float,
              sigma: float, burst_width: float, burst_height: float, epsilon: float, return_dynamics: bool = False):

    # define extrinsic input
    s_ext, noise_lvl, noise_sigma = x[-3:]
    inp = np.zeros((int((T*time_scale+cutoff)/dt),))
    noise = noise_lvl * np.random.randn(inp.shape[0])
    noise = gaussian_filter1d(noise, sigma=noise_sigma)
    inp += s_ext + noise

    # update parameter vector
    for idx, val in zip(x_indices, x):
        func_args[idx] = val
    func_args[inp_idx] = inp

    # simulate model dynamics
    t0 = perf_counter()
    fr = integrate(func, tuple(func_args), T*time_scale + cutoff, dt, dts*time_scale, cutoff) * 1e3
    t1 = perf_counter()
    # print(f"Finished ODE integration after {np.round(t1 - t0, decimals=2)} seconds.")

    # calculate loss based on PSD difference
    if np.isfinite(fr[-1]):
        freqs, fitted_psd = get_psd(fr, fs=1e3/(dts*time_scale), nperseg=nperseg, fmax=fmax, detrend=True)
        burst_stats = get_bursting_stats(fr, sigma=sigma, burst_width=burst_width, rel_burst_height=burst_height,
                                         width_at_height=0.9)
        bursting = np.asarray([burst_stats["ibi_mean"], burst_stats["ibi_std"]])

        loss = mse(y_psd, fitted_psd) + epsilon*mse(y_bursts, bursting)
    else:
        loss = np.inf

    if return_dynamics:
        return loss, fr, freqs, fitted_psd, burst_stats
    return loss


# data set specifics
####################

save_dir = "/home/richard/results"

# choose data set
dataset_name = "trujilo_2019"
path = f"/home/richard-gast/Documents/data/{dataset_name}"
file = "161001"

# choose model
model = "ik_stp"
input_var = "I_ext"

# optimization parameters
n_cpus = 80
maxiter = 100
strategy = "best1exp"
popsize = 80
mutation = (0.5, 1.5)
recombination = 0.6
epsilon = 1e-2
polish = True

# dataset parameters
time_scale = 1000.0
well = 4
well_offset = 4
dt = 1e-2
cutoff = 500.0

# data processing parameters
tau = 20e-3
nperseg = 50000
fmax = 100.0
detrend = True
sigma = 100.0
burst_width = 1000.0
burst_height = 0.5

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
T = time_ds[-1]
dts = float(time_ds[1] - time_ds[0])
spikes = extract_spikes(time, time_ds, spike_times[well + well_offset - 1])
spikes_smoothed = convolve_exp(spikes, tau=tau, dt=dts, normalize=False)
target_fr = np.mean(spikes_smoothed, axis=0) / (tau * time_scale)

# calculate psd
_, target_psd = get_psd(target_fr, fs=1e3/(dts*time_scale), nperseg=nperseg, fmax=fmax, detrend=detrend)

# calculate bursting stats
target_bursts = get_bursting_stats(target_fr, sigma=sigma, burst_width=burst_width, rel_burst_height=burst_height,
                                   width_at_height=0.9)
target_bursts = np.asarray([target_bursts["ibi_mean"], target_bursts["ibi_std"]])

# model initialization
######################

# fixed model parameters
model_params = {
    "v_r": -60.0,
    "v_t": -40.0,
    "eta": 0.0,
    "E_r": 0.0,
    "I_ext": 0.0
}

# initialize model template and set fixed parameters
template = CircuitTemplate.from_yaml(f"config/ik_mf/{model}")
template.update_var(node_vars={f"p/{model}_op/{key}": val for key, val in model_params.items()})

# generate run function
inp = np.zeros((int((T + cutoff)*time_scale/dt),))
func, args, arg_keys, _ = template.get_run_func(f"{model}_vectorfield", step_size=dt,
                                                inputs={f'p/{model}_op/{input_var}': inp},
                                                backend="default", solver="euler")
func_jit = njit(func)
func_jit(*args)

# fitting procedure
###################

# free parameter bounds
bounds = {
    "C": (20.0, 500.0),
    "k": (0.1, 2.0),
    "Delta": (0.01, 2.0),
    "kappa": (0.01, 1.0),
    "f0": (0.2, 1.0),
    "tau_r": (1.0, 20.0),
    "tau_f": (20.0, 400.0),
    "tau_d": (200.0, 2000.0),
    "g": (4.0, 60.0),
    "tau_s": (4.0, 20.0),
    "s_ext": (20.0, 200.0),
    "noise_lvl": (10.0, 100.0),
    "sigma": (5.0, 200.0)
}

# find argument positions of free parameters
param_indices = []
for key in list(bounds.keys())[:-3]:
    idx = arg_keys.index(f"p/{model}_op/{key}")
    param_indices.append(idx)
input_idx = arg_keys.index(f"{input_var}_input_node/{input_var}_input_op/{input_var}_input")

# fitting procedure
func_args = (param_indices, target_psd, target_bursts, func_jit, list(args), input_idx, time_scale, T, dt, dts, cutoff,
             nperseg, fmax, sigma, burst_width, burst_height, epsilon)
while True:
    results = differential_evolution(loss_func, tuple(bounds.values()), args=func_args, strategy=strategy, workers=n_cpus,
                                     disp=True, maxiter=maxiter, popsize=popsize, callback=callback, mutation=mutation,
                                     recombination=recombination, polish=polish)
    if np.isnan(results.fun):
        print("Re-initializing. Reason: loss(best candidate) = NaN.")
    else:
        break
print("And the winner is ... ")
for key, val in zip(bounds.keys(), results.x):
    print(f"{key} = {val}")

# generate dynamics of winner
loss, fr, freqs, psd, bursts = loss_func(results.x, *func_args, return_dynamics=True)

# save results
pickle.dump({"fitting_results": results, "freqs": freqs, "target_fr": target_fr, "target_psd": target_psd,
             "target_bursts": target_bursts, "fitted_fr": fr, "fitted_psd": psd, "fitted_bursts": bursts, "loss": loss},
            open(f"{save_dir}/{dataset_name}/{file}_fitting_results.pkl", "wb"))

# plotting
##########

# fig, axes = plt.subplots(nrows=2, figsize=(12, 8))
# fig.suptitle(f"Fitting results for organoid {date} (age = {age} days)")
#
# # time series
# ax = axes[0]
# ax.plot(time_ds, target_fr, label="target")
# ax.plot(time_ds[:len(fr)], fr, label="fit")
# ax.set_xlabel("time (s)")
# ax.set_ylabel("firing rate (Hz)")
# ax.set_title("Mean-field dynamics")
# ax.legend()
#
# # PSD
# ax = axes[1]
# ax.plot(freqs, target_psd, label="target")
# ax.plot(freqs, psd, label="fit")
# ax.set_xlabel("frequency")
# ax.set_ylabel("log(psd)")
# ax.set_title("Power spectrum")
# ax.legend()
#
# plt.tight_layout()
# plt.show()
