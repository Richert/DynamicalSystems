import pickle
import numpy as np
from typing import Callable
from custom_functions import *
from pyrates import CircuitTemplate
import warnings
import torch
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=RuntimeWarning)
from sbi import utils as utils
from sbi.inference import NPE, simulate_for_sbi
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
import sys


def integrate(inp: np.ndarray, y: np.ndarray, func, args, T, dt, dts, cutoff):

    steps = int(T / dt)
    cutoff_steps = int(cutoff / dt)
    store_step = int(dts / dt)
    store_steps = int((T - cutoff) / dts)
    state_rec = []
    u = float(args[inp_idx])

    # solve ivp for forward Euler method
    for step in range(steps):
        args[inp_idx] = u + inp[step]
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
    inp = np.zeros((int((T + cutoff) / dt),))
    noise = noise_lvl * np.random.randn(inp.shape[0])
    noise = gaussian_filter1d(noise, sigma=noise_sigma)
    inp += noise

    # simulate model dynamics
    fr = integrate(inp, func_args[1], func, func_args[2:], T + cutoff, dt, dts, cutoff)

    # get waveform
    max_idx_model = np.argmax(fr)
    max_idx_target = np.argmax(y_target)
    start = max_idx_model - max_idx_target
    if start < 0:
        start = 0
    if start + waveform_length > fr.shape[0]:
        start = fr.shape[0] - waveform_length
    fr = fr[start:start + waveform_length]

    # fourier transform
    fr_fft = np.fft.rfft(fr)
    fr_psd = np.real(np.abs(fr_fft))

    # return fourier-transformed signal
    if return_dynamics:
        return fr, fr_psd
    return fr_psd

# parameter definitions
#######################

# plotting
plotting = True
save_fig = True
show_fig = True
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 16.0
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['lines.linewidth'] = 1.0
markersize = 40

# choose device
device = "cpu"
n_jobs = 80

# define directories and file to fit
path = "/home/richard"
dataset = "trujilo_2019"
save_dir = f"{path}/results/{dataset}"
load_dir = f"{path}/data/{dataset}"

# choose model
model = "JRC2"
op = "jrc_op"

# choose data set
n_clusters = 5
target_cluster = 1
age = 82
organoid = None

# simulation parameters
cutoff = 1000.0
T = 9000.0
dt = 1e-1
dts = 1.0

# fitting parameters
estimator = "mdn"
n_simulations = int(sys.argv[-2])
n_post_samples = 10000
stop_after_epochs = 30
clip_max_norm = 10.0
lr = 5e-5
n_map_iter = 1000

# choose which SBI steps to run or to load from file
round = int(sys.argv[-1])
uniform_prior = True
run_simulations = True
fit_posterior_model = True

# model parameters
c = 0.1
H_e = 3.25
H_i = 22.0
tau_e = 10.0
tau_i = 20.0
m_max = 1.0
r = 560.0
v_thr = 6.0
ei_ratio = 4.0
noise_lvl = 0.1
noise_sigma = 100.0
u = 0.22
params = {
    'c': c, 'h_e': H_e, 'h_i': H_i, 'tau_e': tau_e, 'tau_i': tau_i, 'm_max': m_max, 'r': r, 'V_thr': v_thr,
    'ei_ratio': ei_ratio, 'u': u,
}

# initialize model template and set fixed parameters
template = CircuitTemplate.from_yaml(f"model_templates.neural_mass_models.jansenrit.{model}")
template.update_var(node_vars={f"jrc/{op}/{key}": val for key, val in params.items()})

# generate run function
func, args, arg_keys, _ = template.get_run_func(f"{model}_vectorfield", step_size=dt, backend="numpy", solver="heun",
                                                float_precision="float64", vectorize=False)

# free parameter bounds
param_bounds = {
    "c": (0.0, 1.0),
    "h_e": (0.1, 20.0),
    "h_i": (0.1, 50.0),
    "tau_e": (1.0, 100.0),
    "tau_i": (5.0, 500.0),
    "r": (50.0, 1000.0),
    "V_thr": (0.0, 20.0),
    "ei_ratio": (0.5, 10.0),
    "u": (0.0, 1.0)
}
param_keys = list(param_bounds.keys())
n_params = len(param_keys)

# find argument positions of free parameters
param_indices = []
for key in param_keys:
    idx = arg_keys.index(f"jrc/{op}/{key}")
    param_indices.append(idx)
inp_idx = arg_keys.index(f"jrc/{op}/u") - 2

# define final arguments of loss/simulation function
func_args = (param_indices, func, list(args), T, dt, dts, cutoff)
simulation_wrapper = lambda theta: simulator(theta.cpu().numpy(), *func_args)

# target data loading and processing
####################################

# load data from file
data = pickle.load(open(f"{load_dir}/{n_clusters}cluster_kmeans_results.pkl", "rb"))
waveforms = data["cluster_centroids"]
y_target = waveforms[target_cluster] / 1e3
waveform_length = len(y_target)

# fourier transform
target_fft = np.fft.rfft(y_target)
target_psd = np.real(np.abs(target_fft))

# fitting procedure
###################

# create priors
if round > 0 and not uniform_prior:

    # load previous model fit
    prior = pickle.load(open(f"{save_dir}/jrc_posterior_n{n_simulations}_p{n_params}_r{round-1}.pkl", "rb"))
    prior.set_default_x(torch.as_tensor(target_psd))

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
check_sbi_inputs(simulation_wrapper, prior)

# create inference object
inference = NPE(prior=prior, density_estimator=estimator, device=device)

# load previously simulated data and append to inference object
if round > 0:
    for r in range(round):
        simulated_data = pickle.load(open(f"{save_dir}/jrc_simulations_n{n_simulations}_p{n_params}_r{r}.pkl", "rb"))
        theta, x = simulated_data["theta"], simulated_data["x"]
        inference = inference.append_simulations(theta, x)

# generate simulations
if run_simulations:

    # simulate data
    theta, x = simulate_for_sbi(simulation_wrapper, proposal=prior, num_simulations=n_simulations,
                                num_workers=n_jobs, show_progress_bar=True)

    # save data
    pickle.dump({"theta": theta, "x": x},
                open(f"{save_dir}/jrc_simulations_n{n_simulations}_p{n_params}_r{round}.pkl", "wb"))

else:

    # load previously simulated data and append to inference object
    simulated_data = pickle.load(
        open(f"{save_dir}/jrc_simulations_n{n_simulations}_p{n_params}_r{round}.pkl", "rb"))
    theta, x = simulated_data["theta"], simulated_data["x"]

# add simulations to inference object
inference = inference.append_simulations(theta, x)

# train the inference object with simulated data
if fit_posterior_model:

    # fit posterior model
    density_estimator = inference.train(stop_after_epochs=stop_after_epochs, clip_max_norm=clip_max_norm,
                                        learning_rate=lr)
    posterior = inference.build_posterior(density_estimator)
    pickle.dump(posterior, open(f"{save_dir}/jrc_posterior_n{n_simulations}_p{n_params}_r{round}.pkl", "wb"))

else:

    # load previous model fit
    posterior = pickle.load(open(f"{save_dir}/jrc_posterior_n{n_simulations}_p{n_params}_r{round}.pkl", "rb"))

# evaluate posterior on target data
###################################

# evaluate posterior
posterior.set_default_x(torch.as_tensor(target_psd))

# generate samples from posterior and create heat map
bins = 50
sampling_obj = density_estimator if fit_posterior_model else posterior
posterior_samples = sampling_obj.sample((n_post_samples,)).numpy()
posterior_grid, x_edges, y_edges = np.histogram2d(x=posterior_samples[:, 0], y=posterior_samples[:, 1],
                                                  bins=bins, density=False)
posterior_grid /= n_post_samples

# get MAP
MAP = posterior.map(num_iter=n_map_iter, num_init_samples=n_post_samples, learning_rate=lr*100, show_progress_bars=True
                    ).numpy().squeeze()

# run the model for the MAP
y_fit, fitted_psd = simulator(MAP, *func_args, return_dynamics=True)

loss = float(np.sum((y_target - y_fit)**2))
print(f"Finished fitting procedure. The MAP parameter set (loss = {loss}) is ... ")
fitted_parameters = {}
for key, val in zip(param_keys, MAP):
    print(f"{key} = {val}")
    fitted_parameters[key] = val

# save results
results = {"target_waveform": y_target, "fitted_waveform": y_fit, "fitted_parameters": fitted_parameters,
           "theta": theta, "x": x}
pickle.dump(results,
            open(f"{save_dir}/ik_organoid_cluster{target_cluster}_fit_n{n_simulations}_p{n_params}_r{round}.pkl", "wb"))

# plotting results
if plotting:
    fig = plt.figure(figsize=(12, 6))
    grid = fig.add_gridspec(ncols=3, nrows=2)
    ax = fig.add_subplot(grid[0, 1:])
    ax.plot(y_target, label="target")
    ax.plot(y_fit, label="fit")
    ax.set_title("Model Dynamics")
    ax.set_xlabel("time")
    ax.set_ylabel("firing rate")
    ax.legend()
    ax = fig.add_subplot(grid[1, 1:])
    freqs = np.fft.rfftfreq(len(y_fit), d=dts*1e-2)
    idx = (freqs > 0.05) * (freqs <= 10.0)
    ax.plot(freqs[idx], target_psd[idx], label="target")
    ax.plot(freqs[idx], fitted_psd[idx], label="fit")
    ax.set_xlabel("frequency")
    ax.set_ylabel("power")
    ax.set_title("Fourier Transform")
    ax.legend()
    ax = fig.add_subplot(grid[:, 0])
    im = ax.imshow(posterior_grid.T, aspect="auto")
    ax.scatter(x=np.argmin((x_edges[:-1] - MAP[0]) ** 2).squeeze(),
               y=np.argmin((y_edges[:-1] - MAP[1]) ** 2).squeeze(),
               marker="x", s=40, color="black", label="MAP")
    ax.legend()
    ax.set_xlabel(param_keys[0])
    ax.set_ylabel(param_keys[1])
    ticks = np.arange(0, bins, step=10, dtype=np.int32)
    ax.set_xticks(ticks, labels=np.round(x_edges[ticks], decimals=2))
    ax.set_yticks(ticks, labels=np.round(y_edges[ticks], decimals=1))
    ax.set_title("Posterior Model")
    plt.tight_layout()
    if save_fig:
        plt.savefig(f"{save_dir}/jrc_organoid_cluster{target_cluster}_fit_n{n_simulations}_p{n_params}_r{round}.png")
    if show_fig:
        plt.show()
