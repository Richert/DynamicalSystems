import pickle
from typing import Callable
import matplotlib.pyplot as plt
from custom_functions import *
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from scipy.ndimage import gaussian_filter
from numba import njit
import torch
from sbi import utils as utils

@njit
def integrate_noise(x, inp, scale, tau):
    return x + scale * inp - x / tau

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
    for sample in range(num_samples):
        x = integrate_noise(x, white_noise[sample], scale, tau)
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

# choose device
device = "cpu"

# define directories and model
path = "/home/richard/data/sbi_organoids"
model = "ik_full"

# simulation parameters
cutoff = 1000.0
dts = 1.0
solver_kwargs = {"atol": 1e-5}

# delay-embedding parameters
nbins = 50
sigma = 1
delays = [2, 4, 8, 16, 32, 64]

# load posterior and model
###########################

# load posterior
posterior = pickle.load(open(f"{path}/{model}_posterior.pkl", "rb"))

# load model
model_info = pickle.load(open(f"{path}/{model}_model.pkl", "rb"))
func = model_info["func"]
args = model_info["args"]
arg_keys = model_info["arg_keys"]
T = model_info["T"]
dt = model_info["dt"]

# find argument positions of free parameters
params = pickle.load(open(f"{path}/{model}_parameters.pkl", "rb"))
param_bounds = params["bounds"]
param_keys = params["parameters"]
prior_min = [param_bounds[key][0] for key in param_keys]
prior_max = [param_bounds[key][1] for key in param_keys]
prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max), device=device)
theta = prior.sample((1,))[0]

param_indices = []
for key in param_keys[:-2]:
    idx = arg_keys.index(f"p/{model}_op/{key}")
    param_indices.append(idx)
inp_idx = arg_keys.index(f"I_ext_input_node/I_ext_input_op/I_ext_input")

# run simulation and evaluate posterior
#######################################

# run simulation
DEs, fr = simulator(theta.cpu().numpy(), param_indices, func, list(args), T, dt, dts, cutoff, return_dynamics=True)

# evaluate posterior at simulated parameter set
x = np.asarray(DEs).flatten()
proposal = posterior.set_default_x(torch.as_tensor(x, device=device, dtype=torch.float32))
theta_fit = proposal.map(num_to_optimize=200)

# print results
for key, t0, t1 in zip(param_keys, theta, theta_fit):
    print(f"{key}: target = {t0:.2f}, fit = {t1:.2f}")

# plot results
##############

fig = plt.figure(figsize=(12, 5))
grid = fig.add_gridspec(nrows=2, ncols=len(DEs))

# plot model dynamics
ax = fig.add_subplot(grid[0, :])
ax.plot(fr)
ax.set_ylabel("r (Hz)")
ax.set_xlabel("time steps")

# plot delay embeddings
for i, DE in enumerate(DEs):
    ax = fig.add_subplot(grid[1, i])
    ax.imshow(DE, cmap="viridis", aspect="auto", interpolation="none")
    ax.set_xticks([])
    ax.set_yticks([])

# save figure
plt.tight_layout()
fig.canvas.draw()
plt.savefig(f"{path}/{model}_fit_synthetic.pdf")
