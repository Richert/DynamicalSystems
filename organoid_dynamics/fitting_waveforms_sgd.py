import pickle
import torch
from typing import Callable
from custom_functions import *
from pyrates import CircuitTemplate
from scipy.ndimage import gaussian_filter1d
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import squareform
import warnings
from time import perf_counter
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=RuntimeWarning)


def integrate(y: torch.Tensor, func, args, T, dt, dts, cutoff, p):

    idx = 0
    steps = int(T / dt)
    cutoff_steps = int(cutoff / dt)
    store_steps = int((T-cutoff) / dts)
    store_step = int(dts / dt)
    state_rec = torch.zeros((store_steps,), dtype=y.dtype, device=y.device)

    # solve ivp for forward Euler method
    for step in range(steps):
        r_e, r_i = y[0], y[6]
        if step > cutoff_steps and step % store_step == t0:
            state_rec[idx] = p*r_e + (1-p)*r_i
            idx += 1
        if not torch.isfinite(r_e):
            break
        rhs = func(step, y, *args)
        y_0 = y + dt * rhs
        y = y + (rhs + func(step, y_0, *args)) * dt/2

    return state_rec


def simulator(x: torch.Tensor, x_indices: list, y: torch.Tensor, func: Callable, func_args: list,
              T: float, dt: float, dts: float, cutoff: float, p: float, waveform_length: int,
              return_dynamics: bool = False):

    # update parameter vector
    idx_half = int(len(x)/2)
    for i, j in enumerate(x_indices):
        func_args[j] = torch.tensor([x[i], x[idx_half + i]], requires_grad=True)

    # simulate model dynamics
    fr = integrate(func_args[1], func, tuple(func_args[2:]), T + cutoff, dt, dts, cutoff, p) * 1e3

    # get waveform
    max_idx_model = torch.argmax(fr)
    max_idx_target = torch.argmax(y)
    start = max_idx_model-max_idx_target
    start = start if start >= 0 else 0
    y_fit = fr[start:start+waveform_length]

    # calculate loss
    loss = torch.sum((y_target - y_fit)**2)

    if return_dynamics:
        return loss, y_fit
    return loss

# parameter definitions
#######################

# torch settings
device = "cpu"
dtype = torch.double

# choose data set
dataset = "trujilo_2019"

# define directories and file to fit
path = "/home/richard-gast/Documents"
save_dir = f"{path}/results/{dataset}"
load_dir = f"{path}/data/{dataset}"

# choose model
model = "ik_eic_sfa"
exc_op = "ik_sfa_op"
inh_op = "ik_sfa_op"

# data processing parameters
tau = 20.0
sigma = 20.0
burst_width = 100.0
burst_sep = 1000.0
burst_height = 0.5
burst_relheight = 0.9
waveform_length = 3000

# torch parameters
lr = 1e-3
tolerance = 1e-2
betas = (0.9, 0.99)
max_epochs = 200

# data loading and processing
#############################

# load data from file
data = pd.read_csv(f"{load_dir}/{dataset}_waveforms.csv", header=[0, 1, 2], index_col=0)
D = np.load(f"{load_dir}/{dataset}_waveform_distances.npy")

# run hierarchical clustering on distance matrix
D_condensed = squareform(D)
Z = linkage(D_condensed, method="ward")
clusters = cut_tree(Z, n_clusters=9)

# extract target waveform
prototype = 2
proto_waves = get_cluster_prototypes(clusters.squeeze(), data, method="random")
y_target = proto_waves[prototype] / np.max(proto_waves[prototype])
plt.plot(y_target)
plt.show()
y_target = torch.tensor(y_target, device=device, dtype=dtype, requires_grad=True)

# model initialization
######################

# simulation parameters
dts = 1.0
dt = 1e-1
cutoff = 1000.0
T = 9000.0 + cutoff
p_e = 0.8 # fraction of excitatory neurons

# exc parameters
exc_params = {
    'C': 100.0, 'k': 0.7, 'v_r': -60.0, 'v_t': -40.0, 'Delta': 2.0, 'eta': 70.0, 'kappa': 10.0, 'tau_u': 500.0,
    'g_e': 150.0, 'g_i': 100.0, 'tau_s': 5.0
}

# inh parameters
inh_params = {
    'C': 150.0, 'k': 0.5, 'v_r': -65.0, 'v_t': -45.0, 'Delta': 4.0, 'eta': 0.0, 'kappa': 20.0, 'tau_u': 100.0,
    'g_e': 60.0, 'g_i': 0.0, 'tau_s': 8.0
}

# initialize model template and set fixed parameters
template = CircuitTemplate.from_yaml(f"config/ik_mf/{model}")
template.update_var(node_vars={f"exc/{exc_op}/{key}": val for key, val in exc_params.items()})
template.update_var(node_vars={f"inh/{inh_op}/{key}": val for key, val in inh_params.items()})

# generate run function
func, args, arg_keys, _ = template.get_run_func(f"{model}_vectorfield", step_size=dt, backend="torch", solver="heun",
                                                float_precision="float64")
grad_args = [f"exc/{exc_op}/{key}" for key in exc_params.keys()]
for key, arg in zip(arg_keys, args):
    if type(arg) is torch.Tensor:
        arg.to(device)
        arg.requires_grad = True if key in grad_args else False
args[1].requires_grad = True

# free parameter bounds
exc_bounds = {
    "Delta": (0.5, 5.0),
    "k": (0.1, 1.0),
    "kappa": (0.0, 40.0),
    "tau_u": (100.0, 1000.0),
    "g_e": (50.0, 200.0),
    "g_i": (20.0, 200.0),
    "eta": (40.0, 100.0),
    "tau_s": (1.0, 10.0),
}
inh_bounds = {
    "Delta": (2.0, 8.0),
    "k": (0.1, 1.0),
    "kappa": (0.0, 40.0),
    "tau_u": (100.0, 1000.0),
    "g_e": (40.0, 150.0),
    "g_i": (0.0, 100.0),
    "eta": (-50.0, 50.0),
    "tau_s": (5.0, 20.0),
}

# find argument positions of free parameters
param_indices = []
for key in list(exc_bounds.keys()):
    idx = arg_keys.index(f"exc/{exc_op}/{key}")
    param_indices.append(idx)

# define final arguments of loss/simulation function
func_args = (param_indices, y_target, func, list(args), T, dt, dts, cutoff, p_e, waveform_length)

# fitting procedure
###################

# combine parameter bounds
x0 = []
bounds = []
param_keys = []
for b, group in zip([exc_bounds, inh_bounds], ["exc", "inh"]):
    for key, (low, high) in b.items():
        p = np.random.rand()
        x0.append((p*low + (1-p)*high))
        bounds.append((low, high))
        param_keys.append(f"{group}/{key}")
x = torch.tensor(x0, device=device, dtype=dtype)

# test run
print(f"Starting a test run of the mean-field model for {np.round(T, decimals=0)} ms simulation time, using a "
      f"simulation step-size of {dt} ms.")
t0 = perf_counter()
with torch.no_grad():
    simulator(x, *func_args, return_dynamics=False)
t1 = perf_counter()
print(f"Finished test run after {t1-t0} s.")

# initialize optimizer
optim = torch.optim.Adam([x], lr=lr, betas=betas)

# fitting procedure
print(f"Starting to fit the mean-field model to {np.round(T, decimals=0)} ms of spike recordings ...")
sse_loss = torch.inf
epoch = 0
with torch.enable_grad():

    while sse_loss > tolerance and epoch < max_epochs:

        loss = simulator(x, *func_args)
        optim.zero_grad()
        loss.backward()
        optim.step()
        sse_loss = loss.item()
        epoch += 1
        print(f"Optimization step {epoch}: loss = {sse_loss}")

# generate dynamics of winner
loss, y_fit = simulator(x, *func_args, return_dynamics=True)
x = x.cpu().detach().numpy()

# get winning parameter set
print(f"Finished fitting procedure. The winner (loss = {loss}) is ... ")
fitted_parameters = {}
for key, val in zip(param_keys, x):
    print(f"{key} = {val}")
    fitted_parameters[key] = val

# save results
pickle.dump({
    "target_waveform": y_target, "fitted_waveform": y_fit, "fitted_parameters": fitted_parameters},
    open(f"{save_dir}/{dataset}_prototype_{prototype}_fit.pkl", "wb"))
