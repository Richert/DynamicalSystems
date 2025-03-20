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


def integrate(y: torch.Tensor, func, args, T, dt, dts, cutoff):

    idx = 0
    steps = int(T / dt)
    cutoff_steps = int(cutoff / dt)
    store_step = int(dts / dt)
    state_rec = []

    # solve ivp for forward Euler method
    for step in range(steps):
        if step > cutoff_steps and step % store_step == 0:
            state_rec.append(y[:2])
            idx += 1
        if not torch.isfinite(y[0]):
            break
        rhs = func(step, y, *args)
        y_0 = y + dt * rhs
        y = y + (rhs + func(step, y_0, *args)) * dt/2

    return torch.stack(state_rec, dim=0)


def simulator(y_target: torch.Tensor, func: Callable, func_args: list,
              T: float, dt: float, dts: float, cutoff: float, p: torch.Tensor, waveform_length: int,
              return_dynamics: bool = False):

    # simulate model dynamics
    fr = integrate(func_args[1], func, func_args[2:], T + cutoff, dt, dts, cutoff) * 1e3
    fr = p*fr[:, 0] + (1-p)*fr[:, 1]

    # get waveform
    max_idx_model = torch.argmax(fr)
    max_idx_target = torch.argmax(y_target)
    start = max_idx_model - max_idx_target
    if start < 0:
        start = 0
    if start + waveform_length > fr.shape[0]:
        start = fr.shape[0] - waveform_length
    y_fit = fr[start:start + waveform_length]
    y_fit = y_fit / torch.max(y_fit)

    # calculate loss
    loss = torch.sum((y_target - y_fit)**2)

    if return_dynamics:
        return loss, y_fit
    return loss

# parameter definitions
#######################

# torch settings
device = "cuda:0"
dtype = torch.double

# choose data set
dataset = "trujilo_2019"

# define directories and file to fit
path = "/home/richard"
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
lr = 5e-2
tolerance = 1e-2
betas = (0.9, 0.99)
max_epochs = 1000

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
# plt.plot(y_target)
# plt.show()

# model initialization
######################

# simulation parameters
dts = 1.0
dt = 1e-1
cutoff = 1000.0
T = 9000.0 + cutoff
p_e = 1.0 # fraction of excitatory neurons

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

# fitting procedure
###################

# test run
print(f"Starting a test run of the mean-field model for {np.round(T, decimals=0)} ms simulation time, using a "
      f"simulation step-size of {dt} ms.")
t0 = perf_counter()
with torch.no_grad():
    target = torch.tensor(y_target, device=device, dtype=dtype, requires_grad=False)
    simulator(target, func, args, T, dt, dts, cutoff, p_e, waveform_length, return_dynamics=False)
t1 = perf_counter()
print(f"Finished test run after {t1-t0} s.")

# initialize optimizer
grad_args = [f"exc/{exc_op}/{key}" for key in exc_params.keys()]
opt_args = []
args = list(args)
for key, arg in zip(arg_keys, args):
    if type(arg) is torch.Tensor:
        arg.to(device)
        arg.requires_grad = False
        if key in grad_args:
            arg.requires_grad = True
            opt_args.append(arg)
args[1].requires_grad = True
p = torch.tensor(p_e, dtype=dtype, device=device, requires_grad=True)
optim = torch.optim.Adam(opt_args + [p], lr=lr, betas=betas)

# fitting procedure
print(f"Starting to fit the mean-field model to the waveform prototype of cluster {prototype} ...")
sse_loss = torch.inf
epoch = 0
with torch.enable_grad():

    while sse_loss > tolerance and epoch < max_epochs:

        # calculate loss and take optimization step
        target = torch.tensor(y_target, device=device, dtype=dtype, requires_grad=False)
        loss, y_fit = simulator(target, func, args, T, dt, dts, cutoff, p_e, waveform_length, return_dynamics=True)
        loss.backward()
        optim.step()

        # reset network state
        optim.zero_grad()
        y_tmp = args[1].detach().cpu().numpy()
        args[1] = torch.tensor(y_tmp, device=device, dtype=dtype, requires_grad=True)
        dy_tmp = args[2].detach().cpu().numpy()
        args[2] = torch.tensor(dy_tmp, device=device, dtype=dtype, requires_grad=False)

        # progress report
        sse_loss = loss.item()
        epoch += 1
        print(f"Optimization step {epoch}: loss = {sse_loss}")

# generate dynamics of winner
with torch.no_grad():
    y = torch.tensor(y_target, device=device, dtype=dtype, requires_grad=False)
    loss, y_fit = simulator(y, func, args, T, dt, dts, cutoff, p_e, waveform_length, return_dynamics=True)

# get winning parameter set
print(f"Finished fitting procedure. The winner (loss = {loss.item()}) is ... ")
fitted_parameters = {}
for key, val in zip(grad_args, opt_args):
    v_fit = val.cpu().detach().numpy()
    exc_key = key
    inh_key = f"inh/{inh_op}/{key.split('/')[-1]}"
    print(f"{exc_key} = {val[0]}")
    print(f"{inh_key} = {val[1]}")
    fitted_parameters[exc_key] = val[0]
    fitted_parameters[inh_key] = val[1]

# save results
pickle.dump({
    "target_waveform": y_target, "fitted_waveform": y_fit.detach().cpu().numpy(), "fitted_parameters": fitted_parameters},
    open(f"{save_dir}/{dataset}_prototype_{prototype}_fit.pkl", "wb"))
