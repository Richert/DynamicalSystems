from rectipy import Network, random_connectivity
from pyrates import CircuitTemplate, NodeTemplate, OperatorTemplate
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import poisson
import sys
from custom_functions import *

# define parameters
###################

# get sweep condition
rep = 0 #int(sys.argv[-1])
g = 0.5 #float(sys.argv[-2])
Delta = 0.0 #float(sys.argv[-3])
ei_ratio = 0.5 #float(sys.argv[-4])
path = "/home/richard-gast/Documents/data" #str(sys.argv[-5])

# meta parameters
device = "cpu"
theta_dist = "gaussian"
epsilon = 1e-2

# rc parameters
alpha = 1e-4
f1 = 9.0
f2 = 14.0
train_window = 60
train_step = 20
sigma = 5
threshold = 0.1
noise_lvl = 0.0 * 1e-3

# general model parameters
N = 1000
E_e = 0.0
E_i = -65.0
v_spike = 50.0
v_reset = -90.0
g_in = 10.0

# input parameters
dt = 1e-2
dts = 1e-1
dur = 20.0
window = 1000.0
n_patterns = 2
p_in = n_patterns/(n_patterns+1)
n_train = 50
n_test = 20
amp = 30.0*1e-3
init_cutoff = 1000.0
inp_cutoff = 100.0

# exc parameters
p_e = 0.8
N_e = int(N*p_e)
C_e = 100.0
k_e = 0.7
v_r_e = -60.0
v_t_e = -40.0
eta_e = 0.0
a_e = 0.03
b_e = -2.0
d_e = 50.0
s_e = 20.0*1e-3
tau_s_e = 6.0

# inh parameters
p_i = 1-p_e
N_i = N-N_e
C_i = 100.0
k_i = 0.7
v_r_i = -60.0
v_t_i = -40.0
eta_i = 0.0
a_i = 0.03
b_i = -2.0
d_i = 50.0
s_i = 20.0*1e-3
tau_s_i = 10.0

# connectivity parameters
p_ee = 0.1
p_ii = 0.4
p_ie = 0.2
p_ei = 0.4
g_ee = ei_ratio * g / np.sqrt(N_e * p_ee)
g_ii = g / np.sqrt(N_i * p_ii)
g_ei = g / np.sqrt(N_i * p_ei)
g_ie = ei_ratio * g / np.sqrt(N_e * p_ie)
W_ee = random_connectivity(N_e, N_e, p_ee, normalize=False)
W_ie = random_connectivity(N_i, N_e, p_ie, normalize=False)
W_ei = random_connectivity(N_e, N_i, p_ei, normalize=False)
W_ii = random_connectivity(N_i, N_i, p_ii, normalize=False)

# define distribution of etas
f = gaussian if theta_dist == "gaussian" else lorentzian
thetas_e = f(N_e, loc=v_t_e, scale=Delta, lb=v_r_e, ub=2*v_t_e-v_r_e)
thetas_i = f(N_i, loc=v_t_i, scale=Delta, lb=v_r_i, ub=2*v_t_i-v_r_i)

# initialize the model
######################

# initialize operators
op = OperatorTemplate.from_yaml("config/ik_snn/ik_op")
exc_vars = {"C": C_e, "k": k_e, "v_r": v_r_e, "v_theta": thetas_e, "eta": eta_e, "tau_u": 1/a_e, "b": b_e, "kappa": d_e,
            "g_e": g_ee, "E_i": E_i, "tau_s": tau_s_e, "v": v_t_e, "g_i": g_ei, "E_e": E_e}
inh_vars = {"C": C_i, "k": k_i, "v_r": v_r_i, "v_theta": thetas_i, "eta": eta_i, "tau_u": 1/a_i, "b": b_i, "kappa": d_i,
            "g_e": g_ie, "E_i": E_i, "tau_s": tau_s_i, "v": v_t_i, "g_i": g_ii, "E_e": E_e}

# initialize E and I network
n = NodeTemplate(name="node", operators=[op])
enet = CircuitTemplate(name="exc", nodes={f"exc_{i}": n for i in range(N_e)})
inet = CircuitTemplate(name="inh", nodes={f"inh_{i}": n for i in range(N_i)})
enet.add_edges_from_matrix(source_var="ik_op/s", target_var="ik_op/s_e",
                           source_nodes=[f"exc_{i}" for i in range(N_e)], weight=W_ee)
inet.add_edges_from_matrix(source_var="ik_op/s", target_var="ik_op/s_i",
                           source_nodes=[f"inh_{i}" for i in range(N_i)], weight=W_ii)
enet.update_var(node_vars={f"all/ik_op/{key}": val for key, val in exc_vars.items()})
inet.update_var(node_vars={f"all/ik_op/{key}": val for key, val in inh_vars.items()})

# combine E and I into single network
eic = CircuitTemplate(name="eic", circuits={"exc": enet, "inh": inet})
eic.add_edges_from_matrix(source_var="ik_op/s", target_var="ik_op/s_e",
                          source_nodes=[f"exc/exc_{i}" for i in range(N_e)],
                          target_nodes=[f"inh/inh_{i}" for i in range(N_i)], weight=W_ie)
eic.add_edges_from_matrix(source_var="ik_op/s", target_var="ik_op/s_i",
                          source_nodes=[f"inh/inh_{i}" for i in range(N_i)],
                          target_nodes=[f"exc/exc_{i}" for i in range(N_e)], weight=W_ei)

# initialize model
net = Network(dt, device=device)
net.add_diffeq_node("eic", eic, input_var="g_e_in", output_var="s", spike_var="spike", reset_var="v",
                    to_file=False, op="ik_op", spike_reset=v_reset, spike_threshold=v_spike, clear=True, N=N)

# simulation 1: steady-state
############################

# define input
T = init_cutoff + inp_cutoff + window
inp = np.zeros((int(T/dt), N))
inp[:, :N_e] += poisson.rvs(mu=s_e*g_in*dt, size=(inp.shape[0], N_e))
inp[:, N_e:] += poisson.rvs(mu=s_i*g_in*dt, size=(inp.shape[0], N_i))
inp = convolve_exp(inp, tau_s_e, dt)

# perform simulation
obs = net.run(inputs=inp[int(inp_cutoff/dt):, :], sampling_steps=int(dts/dt), record_output=True, verbose=False,
              cutoff=int(init_cutoff/dt), enable_grad=False)
s = obs.to_dataframe("out")
s.iloc[:, :N_e] /= tau_s_e
s.iloc[:, N_e:] /= tau_s_i

# calculate dimensionality in the steady-state period
epsilon = 1e-2
s_vals = s.values[:, :N_e]
dim_ss = get_dim(s_vals, center=True) / N_e
s_vals_tmp = s_vals[:, np.mean(s_vals, axis=0)*1e3 > epsilon]
N_ss = s_vals_tmp.shape[1]
dim_ss_r = get_dim(s_vals_tmp, center=True) / N_ss
dim_ss_nc = get_dim(s_vals, center=False) / N_e

# extract spikes in network
spike_counts = []
s_vals = s.values
for idx in range(s_vals.shape[1]):
    peaks, _ = find_peaks(s_vals[:, idx])
    spike_counts.append(peaks)

# calculate firing rate statistics
taus = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0]
ffs, ffs2 = [], []
for tau in taus:
    ffs.append(fano_factor(spike_counts, s_vals.shape[0], int(tau/dts)))
    ffs2.append(fano_factor2(spike_counts, s_vals.shape[0], int(tau/dts)))
s_vals = s.values[:, :N_e]
s_mean = np.mean(s_vals, axis=1)
s_std = np.std(s_vals, axis=1)

# simulation 2: impulse response
################################

# definition of inputs
inp_neurons = np.random.choice(N_e, size=(int(N_e*p_in),))
in_split = int(len(inp_neurons)/n_patterns)
start = int(inp_cutoff/dt)
stop = int((inp_cutoff+dur)/dt)
y0 = {v: val[:] for v, val in net.state.items()}
inputs ={1: lambda: poisson.rvs(mu=amp*g_in*dt, size=(stop-start, in_split)),
         2: lambda: poisson.rvs(mu=amp*g_in*dt, size=(stop-start, in_split))}
noise_e = poisson.rvs(mu=s_e * g_in * dt, size=(int((inp_cutoff + window) / dt), N_e))
noise_i = poisson.rvs(mu=s_i * g_in * dt, size=(int((inp_cutoff + window) / dt), N_i))

# definition of targets
t = np.linspace(0, window*1e-3, int(window/dts))
target_1 = np.sin(2.0*np.pi*f1*t)
target_2 = np.sin(2.0*np.pi*f2*t)
targets = [target_1, target_2]

# collect network responses to stimulation
responses, targets_patrec, targets_funcgen, conditions = [], [], [], []
condition = {i: [] for i in range(n_patterns + 1)}
for trial in range(n_train + n_test):

    # choose input condition
    c = np.random.choice(list(condition.keys()))

    # generate random input for trial
    inp = np.zeros((int((inp_cutoff + window)/dt), N))
    inp[:, :N_e] += noise_e + poisson.rvs(mu=noise_lvl*g_in*dt, size=(int((inp_cutoff + window) / dt), N_e))
    inp[:, N_e:] += noise_i + poisson.rvs(mu=noise_lvl*g_in*dt, size=(int((inp_cutoff + window) / dt), N_i))
    if c > 0:
        inp[start:stop, inp_neurons[(c-1)*in_split:c*in_split]] += inputs[c]()
    inp = convolve_exp(inp, tau_s_e, dt)

    # get network response to input
    obs = net.run(inputs=inp[start:], sampling_steps=int(dts/dt), record_output=True, verbose=False, enable_grad=False)
    ir = obs.to_numpy("out")[:, :N_e] * 1e3 / tau_s_e
    responses.append(ir)
    condition[c].append(ir)

    # generate targets
    target = np.zeros((int(window/dts), n_patterns))
    if c > 0:
        target[:, c-1] = 1.0
    targets_patrec.append(target)
    target = np.zeros((int(window/dts), 1))
    if c > 0:
        target[:, 0] = targets[c-1]
    targets_funcgen.append(target)
    conditions.append(c)

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
targets_patrec = np.asarray(targets_patrec)
targets_funcgen = np.asarray(targets_funcgen)
conditions = np.asarray(conditions)

# postprocessing
################

# calculate trial-averaged network response
impulse_responses = {}
for c in list(condition.keys()):
    ir = np.asarray(condition[c])
    ir_mean = np.mean(ir, axis=0)
    ir_std = np.std(ir, axis=0)
    impulse_responses[c] = {"mean": ir_mean, "std": ir_std}

# calculate the network dimensionality
dim_ir_col, cs = [], []
for r, c in zip(responses[:n_train], conditions[:n_train]):
    if c > 0:
        dim_c = get_dim(r, center=True) / N_e
        dim_nc = get_dim(r, center=False) / N_e
        r_tmp = r[:, np.mean(r, axis=0) > epsilon]
        N_tmp = r_tmp.shape[1]
        dim_r = get_dim(r_tmp, center=True) / N_tmp
        dim_ir_col.append([dim_c, dim_r, dim_nc])
    cs.append(get_cov(r, center=False, alpha=alpha))
dim_irs = np.mean(np.asarray(dim_ir_col), axis=0)

# calculate the network kernel
mean_response = np.mean(responses[:n_train], axis=0)
C_mean = np.mean(cs, axis=0)
C_inv = np.linalg.inv(C_mean)
w = mean_response @ C_inv
K = w @ mean_response.T
G = np.zeros_like(K)
for s_i in responses[:n_train]:
    G += w @ (s_i - mean_response).T
G /= n_train

# calculate the response variance across trials
kernel_var = np.sum(G.flatten()**2)
corr_var = np.sum(np.var(cs, axis=0).flatten())

# calculate the kernel quality
K_shifted = np.zeros_like(K)
for j in range(K.shape[0]):
    K_shifted[j, :] = np.roll(K[j, :], shift=int(K.shape[1] / 2) - j)
K_mean = np.mean(K_shifted, axis=0)
K_var = np.var(K_shifted, axis=0)
K_diag = np.diag(K)

# calculate separability
seps = []
sep = 1.0
for c1 in list(condition.keys()):
    for c2 in list(condition.keys()):
        if c1 != c2:
            sep_tmp = separability(impulse_responses[c1]["mean"], impulse_responses[c2]["mean"], metric="cosine")
            sep *= sep_tmp

# fit dual-exponential to envelope of separability
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
dim_sep, dim_sep_reduced, dim_sep_nc, neuron_dropout = [], [], [], []
for c in condition:
    if c > 0:
        ir = impulse_responses[c]["mean"][:ir_window, :]
        ir_reduced = ir[:, np.mean(ir, axis=0) > epsilon]
        n_neurons = ir_reduced.shape[1]
        dim_sep.append(get_dim(ir, center=True) / N_e)
        dim_sep_reduced.append(get_dim(ir_reduced, center=True) / n_neurons)
        dim_sep_nc.append(get_dim(ir, center=False) / N_e)
        neuron_dropout.append(N_e - n_neurons)
dim_sep = np.mean(dim_sep)
dim_sep_reduced = np.mean(dim_sep_reduced)
dim_sep_nc = np.mean(dim_sep_nc)
neuron_dropout = np.mean(neuron_dropout)

# calculate the prediction performance in the function generation task
funcgen_amp = np.zeros((targets_funcgen.shape[1],))
funcgen_amp[:int(0.9*len(funcgen_amp))] = np.linspace(1.0, 0.0, num=int(0.9*len(funcgen_amp)))
for i in range(targets_funcgen.shape[0]):
    for j in range(targets_funcgen.shape[-1]):
        targets_funcgen[i, :, j] *= funcgen_amp
w_readout = ridge(np.reshape(responses[:n_train], (n_train*int(window/dts), N_e)).T,
                  np.reshape(targets_funcgen[:n_train], (n_train*int(window/dts), 1)).T,
                  alpha=alpha)

funcgen_fit, funcgen_predictions = [], []
for r, t in zip(responses[n_train:], targets_funcgen[n_train:]):
    funcgen_fit.append(K @ t)
    funcgen_predictions.append(r @ w_readout.T)

# calculate the prediction performance in the pattern recognition task
patrec_predictions, patrec_targets = [], []
for i in range(int(window/(dts*train_step))+1):

    # training
    train_responses = np.concatenate([r[i*train_step:i*train_step+train_window, :] for r in responses[:n_train]], axis=0)
    train_targets = np.concatenate([t[i*train_step:i*train_step+train_window, :] for t in targets_patrec[:n_train]], axis=0)
    W_r = ridge(train_responses.T, train_targets.T, alpha=alpha).T

    # testing
    test_responses = np.concatenate([r[i*train_step:i*train_step+train_window, :] for r in responses[n_train:]], axis=0)
    test_targets = np.concatenate([t[i*train_step:i*train_step+train_window, :] for t in targets_patrec[n_train:]], axis=0)
    patrec_predictions.append(test_responses @ W_r)
    patrec_targets.append(test_targets)

# save results
results = {"g": g, "Delta": Delta, "ei_ratio": ei_ratio,
           "s_mean": s_mean, "s_std": s_std, "ff_between": ffs, "ff_within": ffs2, "ff_windows": taus,
           "dim_ss": dim_ss_nc, "dim_ss_reduced": dim_ss_r, "dim_ss_centered": dim_ss,
           "dim_ir": dim_irs[2], "dim_ir_reduced": dim_irs[1], "dim_ir_centered": dim_irs[0],
           "dim_sep": dim_sep_nc, "dim_sep_reduced": dim_sep_reduced, "dim_sep_centered": dim_sep,
           "sep_ir": sep, "fit_ir": ir_fit, "params_ir": params,
           "neuron_dropout": neuron_dropout, "impulse_responses": impulse_responses,
           "funcgen_predictions": funcgen_predictions, "funcgen_targets": targets_funcgen[n_train:],
           "patrec_predictions": patrec_predictions, "patrec_targets": patrec_targets,
           "patrec_lag": train_step*dts, "K_mean": K_mean, "K_var": K_var, "K_diag": K_diag, "G_sum": kernel_var
           }
pickle.dump(results, open(f"{path}/rc_eir{int(10*ei_ratio)}_g{int(10*g)}_D{int(Delta)}_{rep+1}.pkl", "wb"))

# plotting
##########

# # steady-state dynamics
# fig, axes = plt.subplots(nrows=2, figsize=(12, 6))
# ax = axes[0]
# ax.plot(s_mean*1e3, label="mean(r)")
# ax.plot(s_std*1e3, label="std(r)")
# ax.legend()
# ax.set_xlabel("steps")
# ax.set_ylabel("r")
# ax.set_title("Mean-field rate dynamics")
# ax = axes[1]
# im = ax.imshow(s.T, aspect="auto", interpolation="none", cmap="Greys")
# plt.colorbar(im, ax=ax)
# ax.set_xlabel("steps")
# ax.set_ylabel("neurons")
# ax.set_title("Spiking dynamics")
# fig.suptitle(f"D = {dim_ss}, D_r = {dim_ss_r}, D_nc = {dim_ss_nc}")
# plt.tight_layout()
#
# # impulse response dynamics
# fig, axes = plt.subplots(nrows=2, figsize=(12, 5))
# ax = axes[0]
# for c in condition:
#     ax.plot(np.mean(impulse_responses[c]["mean"], axis=1), label=f"input {c}")
# ax.set_xlabel("steps")
# ax.set_ylabel("r (Hz)")
# ax.set_title("Mean-field response")
# ax.legend()
# ax = axes[1]
# ax.plot(sep, label="target IR")
# ax.plot(ir_fit, label="fitted IR")
# ax.legend()
# ax.set_xlabel("steps")
# ax.set_ylabel("S")
# ax.set_title("Input Separability")
# fig.suptitle(f"D = {dim_sep}, D_r = {dim_sep_reduced}, D_nc = {dim_sep_nc}")
# plt.tight_layout()
#
# # network kernels
# fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
# ax = axes[0]
# im = ax.imshow(C_mean, aspect="auto", interpolation="none", cmap="viridis")
# plt.colorbar(im, ax=ax)
# ax.set_xlabel("neurons")
# ax.set_ylabel("neurons")
# ax.set_title("Neural Covariances")
# ax = axes[1]
# im = ax.imshow(K, aspect="auto", interpolation="none", cmap="viridis")
# plt.colorbar(im, ax=ax)
# ax.set_xlabel("steps")
# ax.set_ylabel("steps")
# ax.set_title("Network Response Kernel")
# ax = axes[2]
# im = ax.imshow(G, aspect="auto", interpolation="none", cmap="viridis")
# plt.colorbar(im, ax=ax)
# ax.set_xlabel("steps")
# ax.set_ylabel("steps")
# ax.set_title("Network Response Variance")
# fig.suptitle(f"D = {dim_irs[0]}, D_r = {dim_irs[1]}, D_nc = {dim_irs[2]}")
# plt.tight_layout()
#
# # predictions for pattern recognition task
# loss = np.asarray([np.mean((t-p)**2) for t, p in zip(patrec_targets, patrec_predictions)])
# loss_smoothed = np.asarray(gaussian_filter1d(loss, sigma=sigma))
# try:
#     idx = np.argwhere(loss_smoothed > threshold).squeeze()[0]
# except IndexError:
#     idx = 0
# patrec_predictions = np.concatenate(patrec_predictions)
# patrec_targets = np.concatenate(patrec_targets)
# fig = plt.figure(figsize=(12, n_patterns*3))
# grid = fig.add_gridspec(nrows=n_patterns, ncols=1)
# fig.suptitle("Pattern Recognition Predictions")
# for i in range(n_patterns):
#     ax = fig.add_subplot(grid[i, 0])
#     ax.plot(patrec_targets[:, i], label="target", color="black")
#     ax.plot(patrec_predictions[:, i], label="prediction", color="royalblue")
#     ax.set_xlabel("steps")
#     ax.set_ylabel("readout")
#     ax.set_title(f"Pattern {i+1}")
#     ax.legend()
# plt.tight_layout()
#
# # plotting loss for pattern recognition task
# fig, ax = plt.subplots(figsize=(12, 4))
# fig.suptitle("Pattern Recognition Loss")
# ax.plot(loss)
# ax.plot(loss_smoothed)
# ax.axhline(y=threshold, color="black", linestyle="dashed")
# ax.set_xlabel("lag (ms)")
# ax.set_ylabel("MSE")
# ax.set_title(f"tau = {idx*2.0}")
# plt.tight_layout()
#
# # predictions and fit for function generation task
# n_trials = 5
# fig, axes = plt.subplots(nrows=n_trials, figsize=(12, 2*n_trials))
# fig.suptitle("Function Generation Predictions")
# for i in range(n_trials):
#     trial = np.random.choice(len(conditions[n_train:]))
#     c = conditions[n_train:][trial]
#     ax = axes[i]
#     ax.plot(targets_funcgen[n_train:][trial, :, 0], label="target", color="black")
#     ax.plot(funcgen_fit[trial][:, 0], label="fit", linestyle="dashed", color="royalblue")
#     ax.plot(funcgen_predictions[trial][:, 0], label="prediction", color="darkorange")
#     ax.set_ylabel("readout")
#     ax.set_xlabel("steps")
#     ax.legend()
#     ax.set_title(f"Input condition {c}")
#     ax.set_ylim([-1.1, 1.1])
# plt.tight_layout()
#
# # kernel statistics
# _, axes = plt.subplots(nrows=3, figsize=(12, 9))
# ax = axes[0]
# ax.plot(K_mean)
# ax.set_ylabel("mean(K)")
# ax = axes[1]
# ax.plot(K_var)
# ax.set_ylabel("var(K)")
# ax = axes[2]
# ax.plot(K_diag)
# ax.set_xlabel("steps")
# ax.set_ylabel("diag(K)")
# fig.suptitle("Kernel statistics")
# plt.tight_layout()
#
# plt.show()
