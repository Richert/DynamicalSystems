import numpy as np
from scipy.optimize import least_squares
from kernels import dualexponential
import pickle
from pandas import DataFrame, read_csv
from typing import List, Callable


def residuals(x: np.ndarray, y: np.ndarray, f: Callable, t: np.ndarray):
    """Function for residual calculation.

    :param x: Array containing the parameters of the kernel.
    :param y: Target EPSP.
    :param f: Kernel function.
    :param t: Time vector.
    :return: Residuals between target EPSP and EPSP generated by `f(t, x)`.
    """
    return y - f(t, *tuple(x))


# prepare data
##############

# load data
fn = "dSPN_gabazine"
data = read_csv(f"{fn}.csv")

# cut off irrelevant parts
cutoff_idx = np.argwhere(np.isnan(data.sum(axis=0, skipna=False).values)).squeeze()[0]
data = data.iloc[:, :cutoff_idx]

# bring data into desired format
data = DataFrame(index=data["time"].values, data=data.iloc[:, 1:].values)

# prepare EPSP fitting
######################

# choose form of synaptic response function
func = dualexponential

# parameter names
param_names = ["d", "g", "a", "tau_r", "tau_s", "tau_f"]

# choose initial guess of parameters
tau_r = 60.0
tau_s = 800.0
tau_f = 80.0
a = 0.5
g = 10.0
d = 100.0

p0 = [d, g, a, tau_r, tau_s, tau_f]

# choose parameter boundaries
tau_r_min, tau_r_max = 1.0, 1000.0
tau_s_min, tau_s_max = 10.0, 10000.0
tau_f_min, tau_f_max = 1.0, 1000.0
a_min, a_max = 0.0, 1.0
g_min, g_max = 0.0, 1e2
d_min, d_max = 0.0, 500.0
bounds = ([d_min, g_min, a_min, tau_r_min, tau_s_min, tau_f_min],
          [d_max, g_max, a_max, tau_r_max, tau_s_max, tau_f_max])

# fit each EPSP
###############

fitted_parameters = []
fitted_epsps = []
for idx in range(data.shape[1]):

    # extract time and epsp vectors
    time = data.index.values
    target_epsp = data.iloc[:, idx].values

    # make zero the minimum of EPSP and time
    target_epsp -= np.min(target_epsp)
    time -= np.min(time)

    # fit synaptic response function to EPSP shape
    res = least_squares(residuals, x0=p0, loss="linear", args=(target_epsp, func, time), bounds=bounds, gtol=None)

    # extract fitted parameters
    params = res["x"]

    # generate fitted EPSP shape
    fitted_epsp = func(time, *tuple(params))

    # import matplotlib.pyplot as plt
    # plt.plot(time, target_epsp, color="black", label="target")
    # plt.plot(time, fitted_epsp, color="red", label="fit")
    # plt.title(','.join([f'{p} = {np.round(v, decimals=3)}' for p, v in zip(param_names, params)]))
    # plt.legend()
    # plt.show()

    # save results to lists
    fitted_parameters.append(params)
    fitted_epsps.append(fitted_epsp)

# save results to file
######################

epsps = DataFrame(index=data.index, data=np.asarray(fitted_epsps).T)
params = DataFrame(index=param_names, data=np.asarray(fitted_parameters).T)
with open(f"{fn}.pkl", "wb") as f:
    pickle.dump({"fitted_epsps": epsps, "parameters": params, "target_epsps": data}, f)
    f.close()