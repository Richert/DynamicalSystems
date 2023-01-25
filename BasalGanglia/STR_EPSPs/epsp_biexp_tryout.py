import numpy as np
from scipy.optimize import dual_annealing, Bounds
from kernels import two_biexponential
import matplotlib.pyplot as plt
from pandas import DataFrame
from typing import List, Callable


def text_to_list(text: str, sep: str, remove: List[str] = None, convert_to_float: bool = False) -> list:
    """Function that converts text separated by some separation markers into a list.
    """
    if remove is not None:
        for rem in remove:
            text = text.replace(rem, "")
    if not convert_to_float:
        return text.split(sep)
    return [float(t) for t in text.split(sep)]


def res(x: np.ndarray, y: np.ndarray, f: Callable, t: np.ndarray):
    return y - f(t, *tuple(x))


def rmse(x: np.ndarray, y: np.ndarray, f: Callable, t: np.ndarray) -> float:
    diff = y - f(t, *tuple(x))
    return np.sqrt(diff @ diff)

# prepare data
##############

# read-in file
with open('EPSP-dSPN-22714003.atf', 'rb') as f:
    lines = f.readlines()
    f.close()

# organize data into dataframe
columns = text_to_list(lines[10].decode(), sep="\t", remove=["\r\n", "\""], convert_to_float=False)
data = [text_to_list(line.decode(), sep="\t", remove=["\r\n", "\""], convert_to_float=True) for line in lines[11:]]
df = DataFrame(columns=columns, data=data)

# define relevant meta parameters
n, m = df.shape
epsp_idx = 1
stim_idx = 3

# find stimulation time (time where to start fitting the kernel)
stimulus = df.iloc[:, stim_idx].values
stimulus_diff = np.diff(stimulus)
threshold = 0.5
stim_onset = np.argwhere(np.abs(stimulus_diff) > threshold)[0, 0]

# plot data
###########

fig, axes = plt.subplots(nrows=m-1, figsize=(10, 6))
for i, ax in enumerate(axes):
    ax.plot(df.iloc[:, 0], df.iloc[:, i+1])
    ax.set_xlabel(columns[0])
    ax.set_ylabel(columns[i+1])
plt.tight_layout()
plt.show()

# fit EPSP shape
################

# choose form of synaptic response function
func = two_biexponential

# choose initial guess of parameters
tau_r1 = 0.01
tau_d1 = 0.1
tau_r2 = 0.1
tau_d2 = 1.0
g1 = 50.0
g2 = 10.0

p0 = np.asarray([g1, g2, tau_r1, tau_r2, tau_d1, tau_d2])

# choose parameter boundaries
tau_r1_min, tau_r1_max = 1e-6, 1.0
tau_r2_min, tau_r2_max = 1e-6, 1.0
tau_d1_min, tau_d1_max = 0.01, 5.0
tau_d2_min, tau_d2_max = 0.01, 5.0
g1_min, g1_max = 0.0, 1e4
g2_min, g2_max = 0.0, 1e4
bounds = Bounds(lb=[g1_min, g2_min, tau_r1_min, tau_r2_min, tau_d1_min, tau_d2_min],
                ub=[g1_max, g2_max, tau_r1_max, tau_r2_max, tau_d1_max, tau_d2_max])

# extract data after stimulation time point
time = df.iloc[stim_onset:, 0].values
target_epsp = df.iloc[stim_onset:, epsp_idx].values

# make zero the minimum of EPSP and time
target_epsp -= np.min(target_epsp)
time -= np.min(time)

# fit synaptic response function to EPSP shape
res = dual_annealing(rmse, x0=p0, args=(target_epsp, func, time), bounds=bounds, maxiter=1000, accept=-100)
params = res["x"]

# generate fitted EPSP shape
fitted_epsp = func(time, *tuple(params))

# results plotting
##################

# provide names of fitted parameters
param_names = ["g1", "g2", "tau_r1", "tau_r2", "tau_d1", "tau_d2"]

# plot fitted data against target data
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(time, fitted_epsp, "blue", label="fitted")
ax.plot(time, target_epsp, "orange", label="target")
ax.set_xlabel("time (s)")
ax.set_ylabel("EPSP (mv)")
plt.title(','.join([f'{p} = {np.round(v, decimals=3)}' for p, v in zip(param_names, params)]))
plt.legend()
plt.show()
