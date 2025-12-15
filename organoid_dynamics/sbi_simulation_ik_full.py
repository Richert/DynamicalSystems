import pickle
from typing import Callable
from custom_functions import *
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from scipy.ndimage import gaussian_filter
import sys

def generate_colored_noise(num_samples, tau, scale=1.0):
    """
    Generates Brownian noise by integrating white noise.

    Args:
        num_samples (int): The number of samples in the output Brownian noise.
        scale (float): A scaling factor for the noise amplitude.

    Returns:
        numpy.ndarray: An array containing the generated Brownian noise.
    """
    white_noise = np.random.randn(num_samples)
    x = 0.0
    colored_noise = np.zeros_like(white_noise)
    for sample in range(num_samples):# Generate white noise (Gaussian)
        x += scale*white_noise[sample] - x / tau
        colored_noise[sample] = x
    return colored_noise


def integrate(y: np.ndarray, func, args, T, dt, dts, cutoff):

    steps = int(T / dt)
    cutoff_steps = int(cutoff / dt)
    store_step = int(dts / dt)
    store_steps = int((T - cutoff) / dts)
    state_rec = []

    # solve ivp for forward Euler method
    for step in range(steps):
        if step > cutoff_steps and step % store_step == 0:
            state_rec.append(y[0])
        if not np.isfinite(y[0]):
            n_zeros = store_steps - len(state_rec)
            y0 = 0.0
            state_rec.extend([y0] * n_zeros)
            break
        rhs = func(step, y, *args)
        y_0 = y + dt * rhs
        y = y + (rhs + func(step, y_0, *args)) * dt/2

    return np.asarray(state_rec)


def simulator(x: np.ndarray, x_indices: list, func: Callable, func_args: list,
              T: float, dt: float, dts: float, cutoff: float, return_dynamics: bool = False):

    # update parameter vector
    for i, j in enumerate(x_indices):
        func_args[j] = x[i]

    # random input
    inp = generate_colored_noise(int(T / dt), tau=x[-2], scale=x[-1]*np.sqrt(dt))
    func_args[inp_idx] = np.asarray(inp, dtype=np.float32)

    # simulate model dynamics
    fr = integrate(y=func_args[1], func=func, args=func_args[2:], T=T, dt=dt, dts=dts, cutoff=cutoff) * 1e3

    # calculate delay-embedding of signal
    r_norm = fr / np.max(fr)
    DEs = []
    for d in delays:
        x = r_norm[:-d]
        y = r_norm[d:]
        DE = np.histogram2d(x, y, bins=nbins)[0] + 0.1
        DE = gaussian_filter(np.log(DE), sigma=sigma)
        DEs.append(DE)

    # return fourier-transformed signal
    if return_dynamics:
        return DEs, fr
    return np.asarray(DEs).flatten()

# parameter definitions
#######################

theta_idx = int(sys.argv[-1])

# choose device
device = "cpu"

# define directories
path = "/home/richard/data/sbi_organoids"

# simulation parameters
cutoff = 1000.0
dts = 1.0
solver_kwargs = {"atol": 1e-5}

# delay-embedding parameters
nbins = 50
sigma = 1
delays = [2, 4, 8, 16, 32, 64]

# model parameters
op = "ik_full_op"

# load parameter samples and model
##################################

# load parameters
params = pickle.load(open(f"{path}/ik_full_parameters.pkl", "rb"))
param_keys = params["parameters"]
theta = params["theta"][theta_idx]
n_params = len(param_keys)

# load model
model = pickle.load(open(f"{path}/ik_full_model.pkl", "rb"))
func = model["func"]
args = model["args"]
arg_keys = model["arg_keys"]
T = model["T"]
dt = model["dt"]

# find argument positions of free parameters
param_indices = []
for key in param_keys[:-2]:
    idx = arg_keys.index(f"p/{op}/{key}")
    param_indices.append(idx)
inp_idx = arg_keys.index(f"I_ext_input_node/I_ext_input_op/I_ext_input")

# run simulation
################

x = simulator(theta, param_indices, func, list(args), T, dt, dts, cutoff)
pickle.dump({"x": x, "theta": theta, "parameters": param_keys},
            open(f"{path}/ik_full_results_p{theta_idx}.pkl", "wb"))
