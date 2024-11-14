from rectipy import Network, random_connectivity
from pyrates import OperatorTemplate
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import poisson
import sys
from custom_functions import *

# define parameters
###################

# meta parameters
device = "cpu"
theta_dist = "gaussian"
alpha = 1e-4
epsilon = 1e-2

# general model parameters
N = 500
E_e = 0.0
E_i = -65.0
v_spike = 50.0
v_reset = -90.0
g_in = 10.0
tau_out = 20.0

# get sweep condition
rep = int(sys.argv[-1])
g = float(sys.argv[-2])
Delta = float(sys.argv[-3])
p = float(sys.argv[-4])
path = str(sys.argv[-5])

# input parameters
dt = 1e-2
dts = 1e-1
dur = 20.0
window = 1000.0
n_patterns = 2
p_in = n_patterns/(n_patterns*3)
n_train = 50
n_test = 10
amp = 30.0*1e-3
init_cutoff = 1000.0
inp_cutoff = 100.0

# exc parameters
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
eta = 0.0
a = 0.03
b = -2.0
d = 50.0
s_e = 15.0*1e-3
tau_s = 6.0

# connectivity parameters
g_norm = g / np.sqrt(N * p)
W = random_connectivity(N, N, p, normalize=False)

# define distribution of etas
f = gaussian if theta_dist == "gaussian" else lorentzian
thetas_e = f(N, loc=v_t, scale=Delta, lb=v_r, ub=2 * v_t - v_r)

# initialize the model
######################

# initialize operators
op = OperatorTemplate.from_yaml("config/ik_snn/ik_op")
exc_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": thetas_e, "eta": eta, "tau_u": 1 / a, "b": b, "kappa": d,
            "g_e": g_norm, "E_i": E_i, "tau_s": tau_s, "v": v_t, "g_i": 0.0, "E_e": E_e}

# initialize model
net = Network(dt, device=device)
net.add_diffeq_node("ik", f"config/ik_snn/ik", weights=W, source_var="s", target_var="s_e",
                    input_var="g_e_in", output_var="s", spike_var="spike", reset_var="v", to_file=False,
                    node_vars=exc_vars.copy(), op="ik_op", spike_reset=v_reset, spike_threshold=v_spike,
                    clear=True)

# simulation 1: steady-state
############################

# define input
T = init_cutoff + inp_cutoff + window
inp = np.zeros((int(T/dt), N))
inp += poisson.rvs(mu=s_e*g_in*dt, size=inp.shape)
inp = convolve_exp(inp, tau_s, dt)

# perform steady-state simulation
obs = net.run(inputs=inp[int(inp_cutoff/dt):, :], sampling_steps=int(dts/dt), record_output=True, verbose=False,
              cutoff=int(init_cutoff/dt), enable_grad=False)
s = obs.to_dataframe("out")
s.iloc[:, :] /= tau_s
s_vals = s.values

# calculate dimensionality in the steady-state period
dim_ss = get_dim(s_vals, center=True)
s_vals_tmp = s_vals[:, np.sum(s_vals, axis=0) > 0.0]
dim_ss_r = get_dim(s_vals_tmp, center=True)
dim_ss_nc = get_dim(s_vals, center=False)
N_ss = s_vals_tmp.shape[1]

# extract spikes in network
spike_counts = []
for idx in range(s_vals.shape[1]):
    peaks, _ = find_peaks(s_vals[:, idx])
    spike_counts.append(peaks)

# calculate firing rate statistics
taus = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0]
ffs, ffs2 = [], []
for tau in taus:
    ffs.append(fano_factor(spike_counts, s_vals.shape[0], int(tau/dts)))
    ffs2.append(fano_factor2(spike_counts, s_vals.shape[0], int(tau/dts)))
s_mean = np.mean(s_vals, axis=1)
s_std = np.std(s_vals, axis=1)

# simulation 2: impulse response
################################

# preparations
inp_neurons = np.random.choice(N, size=(int(N*p_in),))
in_split = int(len(inp_neurons)/n_patterns)
start = int(inp_cutoff/dt)
stop = int((inp_cutoff+dur)/dt)
y0 = {v: val[:] for v, val in net.state.items()}
inputs ={1: poisson.rvs(mu=amp*g_in*dt, size=(stop-start, in_split)),
         2: poisson.rvs(mu=amp*g_in*dt, size=(stop-start, in_split))}
noise = poisson.rvs(mu=s_e * g_in * dt, size=(int((inp_cutoff + window) / dt), N))

# collect network responses to stimulation
responses, targets = [], []
condition = {i: [] for i in range(n_patterns + 1)}
for trial in range(n_train + n_test):

    # choose input condition
    c = np.random.choice(list(condition.keys()))

    # generate random input for trial
    inp = np.zeros((int((inp_cutoff + window)/dt), N))
    inp += noise
    if c > 0:
        inp[start:stop, inp_neurons[(c-1)*in_split:c*in_split]] += inputs[c]
    inp = convolve_exp(inp, tau_s, dt)

    # get network response to input
    obs = net.run(inputs=inp[start:], sampling_steps=int(dts/dt), record_output=True, verbose=False, enable_grad=False)
    ir = obs.to_numpy("out") * 1e3 / tau_s
    responses.append(ir)
    condition[c].append(ir)

    # generate target
    target = np.zeros((int(window/dts), n_patterns))
    if c > 0:
        target[:, c-1] = 1.0
    targets.append(target)

    # # test plotting
    # fig, axes = plt.subplots(nrows=2, figsize=(12, 6))
    # ax = axes[0]
    # ax.plot(np.mean(ir, axis=1))
    # ax.set_ylabel("r")
    # ax.set_xlabel("steps")
    # ax.set_title(f"Input condition {c}")
    # ax = axes[1]
    # ax.imshow(inp.T, interpolation="none", cmap="viridis", aspect="auto")
    # ax.set_xlabel("steps")
    # ax.set_ylabel("neurons")
    # plt.tight_layout()
    # plt.show()

    print(f"Finished {trial + 1} of {n_train + n_test} trials.")

responses = np.asarray(responses)
targets = np.asarray(targets)

# processing of simulation results
##################################

# calculate trial-averaged network response
impulse_responses = {}
for c in list(condition.keys()):
    ir = np.asarray(condition[c])
    ir_mean = np.mean(ir, axis=0)
    ir_std = np.std(ir, axis=0)
    impulse_responses[c] = {"mean": ir_mean, "std": ir_std}

# calculate separability
seps = []
sep = 1.0
for c1 in list(condition.keys()):
    for c2 in list(condition.keys()):
        if c1 != c2:
            sep_tmp = separability(impulse_responses[c1]["mean"], impulse_responses[c2]["mean"], metric="cosine")
            sep *= sep_tmp

# fit dual-exponential to envelope of impulse response
tau_r = 10.0
tau_s = 50.0
tau_f = 10.0
scale_s = 0.5
scale_f = 0.5
delay = 5.0
offset = 0.1
p0 = [offset, delay, scale_s, scale_f, tau_r, tau_s, tau_f]
time = s.index.values
time = time - np.min(time)
bounds = ([0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0], [1.0, 100.0, 1.0, 1.0, 1e2, 1e3, 1e2])
params, ir_fit = impulse_response_fit(sep, time, f=dualexponential, bounds=bounds, p0=p0)

# calculate dimensionality in the impulse response period
ir_window = int(2*params[-2]/dts)
dim_irs, dim_irs_reduced, dim_irs_nc, neuron_dropout = [], [], [], []
for c in condition:
    if c > 0:
        ir = impulse_responses[c]["mean"][:ir_window, :]
        ir_reduced = ir[:, np.mean(ir, axis=0) > epsilon]
        dim_irs.append(get_dim(ir, center=True))
        dim_irs_reduced.append(get_dim(ir_reduced, center=True))
        dim_irs_nc.append(get_dim(ir, center=False))
        neuron_dropout.append(ir_reduced.shape[1])

dim_ir = np.mean(dim_irs)
dim_ir_reduced = np.mean(dim_irs_reduced)
dim_ir_nc = np.mean(dim_irs_nc)
neuron_dropout = np.mean(neuron_dropout)

# reservoir computing
train_window = 60
train_step = 20
readout_weights, y_pred, y_targ, loss = [], [], [], []
for i in range(int(window/(dts*train_step))+1):

    # training
    train_responses = np.concatenate([r[i*train_step:i*train_step+train_window, :] for r in responses[:n_train]], axis=0)
    train_targets = np.concatenate([t[i*train_step:i*train_step+train_window, :] for t in targets[:n_train]], axis=0)
    W_r = ridge(train_responses.T, train_targets.T, alpha=alpha).T
    readout_weights.append(W_r)

    # testing
    test_responses = np.concatenate([r[i*train_step:i*train_step+train_window, :] for r in responses[n_train:]], axis=0)
    test_targets = np.concatenate([t[i*train_step:i*train_step+train_window, :] for t in targets[n_train:]], axis=0)
    predictions = test_responses @ W_r
    y_pred.append(predictions)
    y_targ.append(test_targets)
    loss.append(np.mean((test_targets - predictions)**2, axis=0))

# save results
results = {"g": g, "Delta": Delta, "p": p,
           "dim_ss": dim_ss, "s_mean": s_mean, "s_std": s_std, "ff_between": ffs, "ff_within": ffs2, "ff_windows": taus,
           "dim_ir": dim_ir, "sep_ir": sep, "fit_ir": ir_fit, "params_ir": params,
           "impulse_responses": impulse_responses, "predictions": y_pred, "targets": y_targ, "W_out": readout_weights,
           "dim_ir_reduced": dim_ir_reduced, "dim_ss_reduced": dim_ss_r, "dim_ir_nc": dim_ir_nc,
           "dim_ss_nc": dim_ss_nc, "N_ss": N_ss, "N_ir": neuron_dropout, "N": N
           }

# save results
pickle.dump(results, open(f"{path}/rc_exc_g{int(100*g)}_D{int(Delta)}_p{int(100*p)}_{rep+1}.pkl", "wb"))

# preparations for plotting
# y_pred = np.concatenate(y_pred)
# y_targ = np.concatenate(y_targ)
# loss = np.asarray(loss)

# # plotting firing rate dynamics
# fig, ax = plt.subplots(figsize=(12, 4))
# ax.plot(s_mean*1e3, label="mean(r)")
# ax.plot(s_std*1e3, label="std(r)")
# ax.legend()
# ax.set_xlabel("steps")
# ax.set_ylabel("r")
# ax.set_title(f"Dim = {dim_ss}, Dim_r = {dim_ss_r}, dropout = {N - N_ss}")
# fig.suptitle("Mean-field rate dynamics")
# plt.tight_layout()
#
# # plotting spikes
# fig, ax = plt.subplots(figsize=(12, 4))
# im = ax.imshow(s.T, aspect="auto", interpolation="none", cmap="Greys")
# plt.colorbar(im, ax=ax)
# ax.set_xlabel("steps")
# ax.set_ylabel("neurons")
# fig.suptitle("Spiking dynamics")
# plt.tight_layout()
#
# # plotting impulse response
# fig, axes = plt.subplots(nrows=2, figsize=(12, 4))
# fig.suptitle("Impulse response")
# ax = axes[0]
# for c in condition:
#     ax.plot(np.mean(impulse_responses[c]["mean"], axis=1), label=f"input {c}")
# ax.set_xlabel("steps")
# ax.set_ylabel("r (Hz)")
# ax.legend()
# ax = axes[1]
# ax.plot(sep, label="target IR")
# ax.plot(ir_fit, label="fitted IR")
# ax.legend()
# ax.set_xlabel("steps")
# ax.set_ylabel("SR")
# ax.set_title(f"Dim = {np.round(results['dim_ir'], decimals=1)}, tau = {np.round(params[-2], decimals=1)},"
#              f"Dim_r = {np.round(results['dim_ir_reduced'], decimals=1)}")
# plt.tight_layout()
#
# # plotting test data predictions
# fig = plt.figure(figsize=(12, n_patterns*2))
# grid = fig.add_gridspec(nrows=n_patterns, ncols=1)
# fig.suptitle("Test Predictions")
# for i in range(n_patterns):
#     ax = fig.add_subplot(grid[i, 0])
#     ax.plot(y_targ[:, i], label="target", color="black")
#     ax.plot(y_pred[:, i], label="prediction", color="royalblue")
#     ax.set_xlabel("steps")
#     ax.set_ylabel("Out")
#     ax.set_title(f"Pattern {i+1}")
#     ax.legend()
# plt.tight_layout()
#
# # plotting test loss
# fig = plt.figure(figsize=(12, n_patterns*2))
# grid = fig.add_gridspec(nrows=n_patterns, ncols=1)
# fig.suptitle("Test Loss")
# for i in range(n_patterns):
#     ax = fig.add_subplot(grid[i, 0])
#     ax.plot(loss[:, i])
#     ax.set_xlabel("lag (ms)")
#     ax.set_ylabel("MSE")
#     ax.set_title(f"Pattern {i+1}")
#     ax.legend()
# plt.tight_layout()
#
# # plotting fitted weights
# fig = plt.figure(figsize=(12, 8))
# grid = fig.add_gridspec(nrows=2, ncols=1)
# for i in range(n_patterns):
#     ax = fig.add_subplot(grid[i, 0])
#     im = ax.imshow(np.asarray(readout_weights)[:, :, i], aspect="auto", cmap="viridis", interpolation="none")
#     plt.colorbar(im, ax=ax)
#     ax.set_xlabel("neurons")
#     ax.set_ylabel("time lags")
#     ax.set_title(f"Readout for pattern {i+1}")
# fig.suptitle("Readout weights")
# plt.tight_layout()
# plt.show()
