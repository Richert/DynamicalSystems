import pickle
import numpy as np
from typing import Callable
from custom_functions import *
from pyrates import CircuitTemplate
from numba import njit
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
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import squareform
from sklearn.preprocessing import MinMaxScaler
from tslearn.metrics import cdist_dtw

def integrate(y: np.ndarray, func, args, T, dt, dts, cutoff):

    idx = 0
    steps = int(T / dt)
    cutoff_steps = int(cutoff / dt)
    store_step = int(dts / dt)
    store_steps = int((T - cutoff) / dts)
    state_rec = []

    # solve ivp for forward Euler method
    for step in range(steps):
        if step > cutoff_steps and step % store_step == 0:
            state_rec.append(y[0])
            idx += 1
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

    # simulate model dynamics
    fr = integrate(func_args[1], func, func_args[2:], T + cutoff, dt, dts, cutoff) * 1e3

    # fourier transform
    fr_fft = np.fft.rfft(fr)
    fr_psd = np.real(np.abs(fr_fft))

    # return fourier-transformed signal
    if return_dynamics:
        return fr, fr_psd
    return fr_psd

# parameter definitions
#######################

plotting = True

# choose device
device = "cuda:0"
n_jobs = 80

# choose data to fit
dataset = "trujilo_2019"
n_clusters = 4
prototype = 2

# define directories and file to fit
path = "/home/richard"
save_dir = f"{path}/results/{dataset}"
load_dir = f"{path}/data/{dataset}"

# choose model
model = "qif_ca"
op = "qif_ca_op"

# simulation parameters
cutoff = 400.0
T = 1000.0 + cutoff
dt = 5e-3
dts = 1e-1

# fitting parameters
estimator = "mdn"
n_simulations = int(sys.argv[-1])
n_workers = 80
n_post_samples = 1000000
stop_after_epochs = 30
clip_max_norm = 10.0
lr = 5e-5

# data processing parameters
sigma = 20.0
burst_width = 100.0
burst_sep = 1000.0
burst_height = 0.5
burst_relheight = 0.9
waveform_length = 3000

# choose which SBI steps to run or to load from file
run_simulations = True
fit_posterior_model = True

# model parameters
tau = 1.0
Delta = 0.5
eta = 0.2
alpha = 0.4
tau_a = 10.0
a_max = 1.0
a_min = 0.0
J = 40.0
kappa = 0.05
tau_s = 2.0
tau_u = 500.0
I_ext = 0.0
noise_lvl = 30.0
noise_sigma = 100.0
params = {
    'tau': tau, 'Delta': Delta, 'eta': eta, 'alpha': alpha, 'tau_a': tau_a, 'J': J, 'tau_s': tau_s,
    'tau_u': tau_u, 'kappa': kappa, 'a_max': a_max, 'a_min': a_min
}

# free parameter bounds
param_bounds = {
    "tau": (0.5, 5.0),
    "Delta": (0.1, 10.0),
    "eta": (-10.0, 10.0),
    "alpha": (0.0, 1.0),
    "tau_a": (5.0, 50.0),
    "kappa": (0.0, 1.0),
    "tau_u": (50.0, 500.0),
    "J": (1.0, 100.0),
    "tau_s": (0.1, 5.0),
    # "a_max": (0.5, 1.0),
    "a_min": (0.0, 0.8)
}
param_keys = list(param_bounds.keys())
n_params = len(param_keys)

# model initialization
######################

# initialize model template and set fixed parameters
template = CircuitTemplate.from_yaml(f"config/ik_mf/{model}")
template.update_var(node_vars={f"p/{op}/{key}": val for key, val in params.items()})

# generate run function
inp = np.zeros((int((T + cutoff)/dt),))
func, args, arg_keys, _ = template.get_run_func(f"{model}_vectorfield", step_size=dt, backend="numpy", solver="heun",
                                                float_precision="float64", vectorize=False)
func_jit = njit(func)
func_jit(*args)

# find argument positions of free parameters
param_indices = []
for key in param_keys:
    idx = arg_keys.index(f"p/{op}/{key}")
    param_indices.append(idx)

# define final arguments of loss/simulation function
func_args = (param_indices, func_jit, list(args), T, dt, dts, cutoff)
simulation_wrapper = lambda theta: simulator(theta.cpu().numpy(), *func_args)

# data-free fitting procedure (SBI)
###################################

# create priors
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

# generate simulations and pass to the inference object
if run_simulations:
    theta, x = simulate_for_sbi(simulation_wrapper, proposal=prior, num_simulations=n_simulations, num_workers=n_workers)
    pickle.dump({"theta": theta, "x": x}, open(f"{path}/qif_ca_simulations_n{n_simulations}_p{n_params}.pkl", "wb"))
else:
    simulated_data = pickle.load(open(f"{path}/qif_ca_simulations_n{n_simulations}_p{n_params}.pkl", "rb"))
    theta, x = simulated_data["theta"], simulated_data["x"]
inference = inference.append_simulations(theta, x)

# train the density estimator and build the posterior
if fit_posterior_model:
    density_estimator = inference.train(stop_after_epochs=stop_after_epochs, clip_max_norm=clip_max_norm,
                                        learning_rate=lr)
    posterior = inference.build_posterior(density_estimator)
    pickle.dump(posterior, open(f"{path}/qif_ca_posterior_n{n_simulations}_p{n_params}.pkl", "wb"))
else:
    posterior = pickle.load(open(f"{path}/qif_ca_posterior_n{n_simulations}_p{n_params}.pkl", "rb"))

# data loading and processing
#############################

# load data from file
data = pd.read_csv(f"{load_dir}/{dataset}_waveforms.csv", header=[0, 1, 2], index_col=0)

# reduce data
age = 82
organoid = None
normalize = True
data = reduce_df(data, age=age, organoid=organoid)
data_norm = data.values.T
if normalize:
    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(data_norm)
D = cdist_dtw(data_norm[:, :], n_jobs=n_jobs)

# run hierarchical clustering on distance matrix
D_condensed = squareform(D)
Z = linkage(D_condensed, method="ward")
clusters = cut_tree(Z, n_clusters=n_clusters)

# extract target waveform
proto_waves = get_cluster_prototypes(clusters.squeeze(), data, reduction_method="mean")
y_target = proto_waves[prototype]

# fourier transform
target_fft = np.fft.rfft(y_target)
target_psd = np.real(np.abs(target_fft))

# evaluate posterior on target data
###################################

# evaluate posterior
posterior.set_default_x(torch.as_tensor(target_psd))

# generate samples from posterior and create heat map
bins = 50
posterior_samples = posterior.sample((n_post_samples,)).numpy()
posterior_grid, x_edges, y_edges = np.histogram2d(x=posterior_samples[:, 0], y=posterior_samples[:, 1],
                                                  bins=bins, density=False)
posterior_grid /= n_post_samples

# get MAP
MAP = np.unravel_index(np.argmax(posterior_grid), posterior_grid.shape)
x_map = (x_edges[MAP[0]] + x_edges[MAP[0]+1])/2
y_map = (y_edges[MAP[1]] + y_edges[MAP[1]+1])/2
MAP = np.asarray([x_map, y_map])

# run the model for the MAP
y_fit, fitted_psd = simulator(MAP, *func_args, return_dynamics=True)

loss = float(np.sum((y_target - y_fit)**2))
print(f"Finished fitting procedure. The MAP parameter set (loss = {loss}) is ... ")
fitted_parameters = {}
for key, val in zip(param_keys, MAP):
    print(f"{key} = {val}")
    fitted_parameters[key] = val

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
    ax.legend()
    ax.set_xlabel(param_keys[0])
    ax.set_ylabel(param_keys[1])
    ticks = np.arange(0, bins, step=10, dtype=np.int32)
    ax.set_xticks(ticks, labels=np.round(x_edges[ticks], decimals=2))
    ax.set_yticks(ticks, labels=np.round(y_edges[ticks], decimals=1))
    ax.set_title("Posterior Model")
    plt.tight_layout()
    plt.show()

# save results
pickle.dump({
    "target_waveform": y_target, "fitted_waveform": y_fit, "fitted_parameters": fitted_parameters,
    "theta": theta, "x": x},
    open(f"{path}/organoid_fit_n{n_simulations}_p{n_params}.pkl", "wb"))
