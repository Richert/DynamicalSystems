import numpy as np
from scipy.optimize import least_squares, Bounds
from kernels import dualexponential
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
func = dualexponential

# choose initial guess of parameters
tau_r = 0.06
tau_s = 0.8
tau_f = 0.08
a = 0.5
g = 100.0

p0 = [g, a, tau_r, tau_s, tau_f]

# choose parameter boundaries
tau_r_min, tau_r_max = 1e-6, 1.0
tau_s_min, tau_s_max = 0.1, 5.0
tau_f_min, tau_f_max = 0.01, 1.0
a_min, a_max = 0.0, 1.0
g_min, g_max = 0.0, 1e4
bounds = ([g_min, a_min, tau_r_min, tau_s_min, tau_f_min],
          [g_max, a_max, tau_r_max, tau_s_max, tau_f_max])

# extract data after stimulation time point
time = df.iloc[stim_onset:, 0].values
target_epsp = df.iloc[stim_onset:, epsp_idx].values

# make zero the minimum of EPSP and time
target_epsp -= np.min(target_epsp)
time -= np.min(time)

# fit synaptic response function to EPSP shape
res = least_squares(res, x0=p0, loss="soft_l1", args=(target_epsp, func, time), bounds=bounds, gtol=None)
params = res["x"]

# generate fitted EPSP shape
fitted_epsp = func(time, *tuple(params))

# results plotting
##################

# provide names of fitted parameters
param_names = ["g", "a", "tau_r", "tau_s", "tau_f"]

# plot fitted data against target data
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(time, fitted_epsp, "blue", label="fitted")
ax.plot(time, target_epsp, "orange", label="target")
ax.set_xlabel("time (s)")
ax.set_ylabel("EPSP (mv)")
plt.title(','.join([f'{p} = {np.round(v, decimals=3)}' for p, v in zip(param_names, params)]))
plt.legend()
plt.show()
