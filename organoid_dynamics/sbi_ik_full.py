import pickle
from typing import Callable
from custom_functions import *
from pyrates import CircuitTemplate
from numba import njit
import warnings
import torch
warnings.filterwarnings("ignore", category=RuntimeWarning)
from sbi import utils as utils
from sbi.inference import NPE, simulate_for_sbi
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
import sys
from scipy.ndimage import gaussian_filter
from scipy.integrate import solve_ivp

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


def simulator(x: np.ndarray, x_indices: list, func: Callable, func_args: list,
              T: float, dt: float, dts: float, cutoff: float, return_dynamics: bool = False):

    # update parameter vector
    for i, j in enumerate(x_indices):
        func_args[j] = x[i]

    # random input
    inp = generate_colored_noise(int(T / dt), tau=x[-2], scale=x[-1]*np.sqrt(dt))
    func_args[inp_idx] = np.asarray(inp, dtype=np.float32)

    # simulate model dynamics
    res = solve_ivp(fun=func, t_span=(0.0, T), y0=func_args[1], first_step=dt, args=tuple(func_args[2:]),
                    t_eval=np.arange(cutoff, T, dts), **solver_kwargs)
    fr = res.y[0, :] * 1e3

    # calculate delay-embedding of signal
    r_norm = fr / np.max(fr)
    DEs = []
    for d in delays:
        x = r_norm[:-d]
        y = r_norm[d:]
        DE = np.histogram2d(x, y, bins=nbins)[0] + 0.1
        DE = gaussian_filter(np.log(DE), sigma=sigma)
        DEs.append(DE)

    # # get first-level IMFs from firing rates
    # sr = int(dts/dt)
    # imf = emd.sift.mask_sift(fr, max_imfs=max_imfs_lvl1)
    # IP, IF, IA = emd.spectra.frequency_transform(imf, sr, 'hilbert')
    #
    # # get second-level IMFs for the phase envelope of the first-levek IMFs
    # masks = np.array([mask_freq / 2**ii for ii in range(n_masks)]) / sr
    # config = emd.sift.get_config('mask_sift')
    # config['max_imfs'] = max_imfs_lvl2
    # config['imf_opts/sd_thresh'] = imf_thresh
    # imf2 = emd.sift.mask_sift_second_layer(IA, masks, sift_args=config)
    # IP2, IF2, IA2 = emd.spectra.frequency_transform(imf2, sr, 'hilbert')
    #
    # # Compute the 1d Hilbert-Huang transform (power over carrier frequency)
    # _, spec = emd.spectra.hilberthuang(IF, IA, carrier_hist, sum_imfs=False)
    #
    # # Compute the 3d Holospectrum transform (power over time x carrier frequency x AM frequency)
    # fcarrier, fam, holo = emd.spectra.holospectrum(IF, IF2, IA2, carrier_hist, am_hist)
    # sholo = gaussian_filter(holo, 1)

    # return fourier-transformed signal
    if return_dynamics:
        return DEs, fr #fcarrier, spec, fam, sholo,
    return np.asarray(DEs).flatten()

# parameter definitions
#######################

# choose device
device = "cpu"
n_jobs = 15

# define directories
path = "/home/richard/data/sbi_organoids"

# choose model
model = "pc"
op = "ik_full_op"

# simulation parameters
T = 60000.0
cutoff = 0.0
dt = 1e-3
dts = 1.0
solver_kwargs = {}

# sbi parameters
estimator = "mdn"
round = int(sys.argv[-1])
n_simulations = int(sys.argv[-2])
n_workers = 15
n_post_samples = 10000
stop_after_epochs = 30
clip_max_norm = 10.0
lr = 5e-5
n_map_iter = 1000

# delay-embedding parameters
nbins = 50
sigma = 1
delays = [5, 10, 20, 40, 80]

# EMD parameters
max_imfs_lvl1 = 7
max_imfs_lvl2 = 5
imf_thresh = 0.05
mask_freq = 24.0
n_masks = 8
carrier_hist = (1, 20, 128, 'log')
am_hist = (1e-3, 10, 64, 'log')

# model parameters
C = 100.0
k = 0.7
v_r = -70.0
v_t = -45.0
Delta = 1.0
eta = 85.0
b = -2.0
kappa = 5.0
U0 = 0.6
alpha = 0.0
psi = 300.0
theta = 0.02
g_a = 30.0
g_n = 0.1
g_g = 0.0
tau_w = 50.0
tau_ca = 250.0
tau_u = 100.0
tau_x = 700.0
tau_a = 5.0
tau_n = 150.0
tau_g = 10.0
tau_s = 1.0
node_vars = {
    'C': C, 'k': k, 'v_r': v_r, 'v_t': v_t, 'Delta': Delta, 'eta': eta, 'kappa': kappa, 'alpha': alpha,
    'g_a': g_a, 'g_n': g_n, 'g_g': g_g, 'b': b, 'U0': U0, 'tau_ca': tau_ca, 'tau_w': tau_w, 'tau_u': tau_u,
    'tau_x': tau_x, 'tau_a': tau_a, 'tau_n': tau_n, 'tau_g': tau_g, 'tau_s': tau_s, 'psi': psi, 'theta': theta
}

# initialize model template and set fixed parameters
template = CircuitTemplate.from_yaml(f"config/ik_mf/{model}")
template.update_var(node_vars={f"p/{op}/{key}": val for key, val in node_vars.items()})

# generate run function
func, args, arg_keys, _ = template.get_run_func(f"{model}_vectorfield", step_size=dt, backend="numpy", solver="scipy",
                                                float_precision="float32", vectorize=False,
                                                inputs={f"p/{op}/I_ext": np.zeros(int(T/dt),)})
func_jit = njit(func)
func_jit(*args)

# free parameter bounds
param_bounds = {
    "C": (50.0, 200.0),
    "k": (0.5, 1.5),
    "Delta": (0.01, 10.0),
    "eta": (-5.0, 200.0),
    "kappa": (0.0, 100.0),
    "alpha": (0.0, 1.0),
    "g_a": (0.0, 100.0),
    "g_n": (0.0, 10.0),
    "b": (-10.0, 5.0),
    "U0": (0.0, 1.0),
    "tau_ca": (50.0, 500.0),
    "tau_w": (10.0, 100.0),
    "tau_u": (20.0, 200.0),
    "tau_x": (200.0, 2000.0),
    "tau_a": (1.0, 10.0),
    "tau_n": (15.0, 150.0),
    "tau_s": (0.5, 5.0),
    "psi": (50.0, 500.0),
    "theta": (0.0, 0.1),
    "noise_tau": (2.0, 200.0),
    "noise_scale": (0.0, 10.0)
}
param_keys = list(param_bounds.keys())
n_params = len(param_keys)

# find argument positions of free parameters
param_indices = []
for key in param_keys[:-2]:
    idx = arg_keys.index(f"p/{op}/{key}")
    param_indices.append(idx)
inp_idx = arg_keys.index(f"I_ext_input_node/I_ext_input_op/I_ext_input")

# define final arguments of loss/simulation function
func_args = (param_indices, func_jit, list(args), T, dt, dts, cutoff)
simulation_wrapper = lambda theta: simulator(theta.cpu().numpy(), *func_args)

# fitting procedure
###################

# create priors
if round > 0:

    # load previous model fit
    prior = pickle.load(open(f"{path}/ik_ca_posterior_n{n_simulations}_p{n_params}_r{round-1}.pkl", "rb"))

else:

    # create uniform prior
    prior_min = [param_bounds[key][0] for key in param_keys]
    prior_max = [param_bounds[key][1] for key in param_keys]
    prior = utils.torchutils.BoxUniform(
        low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max)
    )

# Check prior, simulator, consistency
prior, num_parameters, prior_returns_numpy = process_prior(prior)
simulation_wrapper = process_simulator(simulation_wrapper, prior, prior_returns_numpy)
# check_sbi_inputs(simulation_wrapper, prior)

# create inference object
inference = NPE(prior=prior, density_estimator=estimator, device=device)

# load previously simulated data and append to inference object
if round > 0:
    for r in range(round):
        simulated_data = pickle.load(open(f"{path}/qif_ca_simulations_n{n_simulations}_p{n_params}_r{r}.pkl", "rb"))
        theta, x = simulated_data["theta"], simulated_data["x"]
        inference = inference.append_simulations(theta, x)

# simulate data
theta, x = simulate_for_sbi(simulation_wrapper, proposal=prior, num_simulations=n_simulations, num_workers=n_workers,
                            show_progress_bar=True)

# fit posterior model
inference = inference.append_simulations(theta, x)
density_estimator = inference.train(stop_after_epochs=stop_after_epochs, clip_max_norm=clip_max_norm,
                                    learning_rate=lr)
posterior = inference.build_posterior(density_estimator)

# save data
pickle.dump({"theta": theta, "x": x},
            open(f"{path}/ik_full_simulations_n{n_simulations}_p{n_params}_r{round}.pkl", "wb"))
pickle.dump(posterior, open(f"{path}/ik_ca_posterior_n{n_simulations}_p{n_params}_r{round}.pkl", "wb"))
