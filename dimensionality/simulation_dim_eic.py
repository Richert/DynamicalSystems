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
rep = int(sys.argv[-1])
g = float(sys.argv[-2])
Delta = float(sys.argv[-3])
ei_ratio = float(sys.argv[-4])
path = str(sys.argv[-5])

# meta parameters
device = "cpu"
theta_dist = "gaussian"
epsilon = 1e-2
alpha = 1e-4

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
n_patterns = 1
p_in = 0.2
n_trials = 10
amp = 50.0 * 1e-3
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
d_e = 100.0
s_e = 20.0*1e-3
tau_s_e = 6.0

# inh parameters
p_i = 1-p_e
N_i = N-N_e
C_i = 20.0
k_i = 1.0
v_r_i = -65.0
v_t_i = -40.0
eta_i = 0.0
a_i = 0.2
b_i = 2.0
d_i = 0.0
s_i = 20.0*1e-3
tau_s_i = 10.0

# connectivity parameters
p_ee = 0.2
p_ii = 0.4
p_ie = 0.2
p_ei = 0.4
sigma_ee = 0.2
sigma_ii = 0.2
sigma_ie = 0.4
sigma_ei = 0.6
g_ee = ei_ratio * g / np.sqrt(N_e * p_ee)
g_ii = g / np.sqrt(N_i * p_ii)
g_ei = g / np.sqrt(N_i * p_ei)
g_ie = ei_ratio * g / np.sqrt(N_e * p_ie)

# define connectivity
W_ee = circular_connectivity(N_e, N_e, p_ee, homogeneous_weights=False, dist="gaussian", scale=sigma_ee)
W_ie = circular_connectivity(N_i, N_e, p_ie, homogeneous_weights=False, dist="gaussian", scale=sigma_ie)
W_ei = circular_connectivity(N_e, N_i, p_ei, homogeneous_weights=False, dist="gaussian", scale=sigma_ei)
W_ii = circular_connectivity(N_i, N_i, p_ii, homogeneous_weights=False, dist="gaussian", scale=sigma_ii)

# fig, ax = plt.subplots(figsize=(6, 6))
# im = ax.imshow(W_ie, interpolation="none", cmap="cividis", aspect="auto")
# plt.colorbar(im, ax=ax, shrink=0.7)
# plt.tight_layout()
# plt.show()

# define distribution of etas
f = gaussian if theta_dist == "gaussian" else lorentzian
thetas_e = f(N_e, loc=v_t_e, scale=Delta, lb=v_r_e, ub=2*v_t_e-v_r_e)
thetas_i = f(N_i, loc=v_t_i, scale=Delta, lb=v_r_i, ub=2*v_t_i-v_r_i)

# initialize the model10
######################

# initialize operators
op_key = "ik_op"
op = OperatorTemplate.from_yaml(f"config/ik_snn/{op_key}")
exc_vars = {"C": C_e, "k": k_e, "v_r": v_r_e, "v_theta": thetas_e, "eta": eta_e, "tau_u": 1/a_e, "b": b_e, "kappa": d_e,
            "g_e": g_ee, "E_i": E_i, "tau_s": tau_s_e, "v": v_t_e, "g_i": g_ei, "E_e": E_e,
            # "tau_d": tau_d_e, "alpha": alpha_e, "tau_f": tau_f_e, "F0": beta_e
            }
inh_vars = {"C": C_i, "k": k_i, "v_r": v_r_i, "v_theta": thetas_i, "eta": eta_i, "tau_u": 1/a_i, "b": b_i, "kappa": d_i,
            "g_e": g_ie, "E_i": E_i, "tau_s": tau_s_i, "v": v_t_i, "g_i": g_ii, "E_e": E_e,
            # "tau_d": tau_d_i, "alpha": alpha_i, "tau_f": tau_f_i, "F0": beta_i
            }

# initialize E and I network
n = NodeTemplate(name="node", operators=[op])
enet = CircuitTemplate(name="exc", nodes={f"exc_{i}": n for i in range(N_e)})
inet = CircuitTemplate(name="inh", nodes={f"inh_{i}": n for i in range(N_i)})
enet.add_edges_from_matrix(source_var=f"{op_key}/s", target_var=f"{op_key}/s_e",
                           source_nodes=[f"exc_{i}" for i in range(N_e)], weight=W_ee)
inet.add_edges_from_matrix(source_var=f"{op_key}/s", target_var=f"{op_key}/s_i",
                           source_nodes=[f"inh_{i}" for i in range(N_i)], weight=W_ii)
enet.update_var(node_vars={f"all/{op_key}/{key}": val for key, val in exc_vars.items()})
inet.update_var(node_vars={f"all/{op_key}/{key}": val for key, val in inh_vars.items()})

# combine E and I into single network
eic = CircuitTemplate(name="eic", circuits={"exc": enet, "inh": inet})
eic.add_edges_from_matrix(source_var=f"{op_key}/s", target_var=f"{op_key}/s_e",
                          source_nodes=[f"exc/exc_{i}" for i in range(N_e)],
                          target_nodes=[f"inh/inh_{i}" for i in range(N_i)], weight=W_ie)
eic.add_edges_from_matrix(source_var=f"{op_key}/s", target_var=f"{op_key}/s_i",
                          source_nodes=[f"inh/inh_{i}" for i in range(N_i)],
                          target_nodes=[f"exc/exc_{i}" for i in range(N_e)], weight=W_ei)

# initialize model
net = Network(dt, device=device)
net.add_diffeq_node("eic", eic, input_var="g_e_in", output_var="s", spike_var="spike", reset_var="v",
                    to_file=False, op=op_key, spike_reset=v_reset, spike_threshold=v_spike, clear=True, N=N)

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
s_vals = s.values[:, :N_e]
s_vals_tmp = s_vals[:, np.mean(s_vals, axis=0)*1e3 > epsilon]
N_ss = s_vals_tmp.shape[1]
dim_ss = get_dim(s_vals, center=False) / N_e
dim_ss_r = get_dim(s_vals_tmp, center=False) / N_ss
dim_ss_c = get_dim(s_vals, center=True) / N_e
dim_ss_rc = get_dim(s_vals_tmp, center=True) / N_ss

# extract spikes in network
spike_counts = []
s_vals = s.values
for idx in range(s_vals.shape[1]):
    peaks, _ = find_peaks(s_vals[:, idx])
    spike_counts.append(peaks)

# calculate firing rate statistics
taus = [50.0, 100.0, 200.0]
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
start = int(inp_cutoff/dt)
stop = int((inp_cutoff+dur)/dt)
W_in = input_connectivity(np.linspace(-np.pi, np.pi, num=n_patterns),
                          N_e, p_in, homogeneous_weights=False, dist="gaussian", scale=0.5)
impulse = poisson.rvs(mu=amp*g_in*dt, size=(stop-start, 1)) @ W_in

# collect network responses to stimulation
responses = {i: [] for i in range(n_patterns + 1)}
trial = 0
while trial < n_trials:

    # define background input
    background_e = poisson.rvs(mu=s_e * g_in * dt, size=(int((inp_cutoff + window) / dt), N_e))
    background_i = poisson.rvs(mu=s_i * g_in * dt, size=(int((inp_cutoff + window) / dt), N_i))

    for c in np.random.permutation(n_patterns+1):

        # generate random input for trial
        inp = np.zeros((int((inp_cutoff + window)/dt), N))
        inp[:, :N_e] += background_e
        inp[:, N_e:] += background_i
        if c > 0:
            inp[start:stop, :N_e] += impulse
        inp = convolve_exp(inp, tau_s_e, dt)

        # get network response to input
        obs = net.run(inputs=inp[start:], sampling_steps=int(dts/dt), record_output=True, verbose=False, enable_grad=False)
        ir = obs.to_numpy("out")[:, :N_e] * 1e3 / tau_s_e

        # save trial results
        responses[c].append(ir)

        # # test plotting
        # fig, axes = plt.subplots(nrows=3, figsize=(12, 8))
        # ax = axes[0]
        # ax.plot(np.mean(ir, axis=1))
        # ax.set_ylabel("r")
        # ax.set_title(f"Input condition {c}")
        # ax = axes[1]
        # ax.imshow(ir.T, interpolation="none", cmap="Greys", aspect="auto")
        # ax.set_ylabel("neurons")
        # ax.set_title("spikes")
        # ax = axes[2]
        # im = ax.imshow(inp.T, interpolation="none", cmap="viridis", aspect="auto")
        # ax.set_xlabel("steps")
        # ax.set_ylabel("neurons")
        # ax.set_title("input")
        # plt.colorbar(im, ax=ax, shrink=0.7)
        # plt.tight_layout()
        # plt.show()

    trial += 1

    # print(f"Finished {trial} of {n_trials} trials.")

# postprocessing
################

# calculate trial-averaged network response
impulse_responses = {}
for c in list(responses.keys()):
    ir = np.asarray(responses[c])
    ir_mean = np.mean(ir, axis=0)
    ir_std = np.std(ir, axis=0)
    impulse_responses[c] = {"mean": ir_mean, "std": ir_std}

# calculate the network dimensionality
dim_ir_col, cs = [], []
for r in responses[1]:
    dim_c = get_dim(r, center=True) / N_e
    dim_nc = get_dim(r, center=False) / N_e
    r_tmp = r[:, np.mean(r, axis=0) > epsilon]
    N_tmp = r_tmp.shape[1]
    dim_r = get_dim(r_tmp, center=False) / N_tmp
    dim_rc = get_dim(r_tmp, center=True) / N_tmp
    dim_ir_col.append([dim_nc, dim_r, dim_c, dim_rc])
    cs.append(get_cov(r, center=False, alpha=alpha))
dim_irs = np.mean(np.asarray(dim_ir_col), axis=0)

# calculate the network kernel
mean_response = np.mean(responses[1], axis=0)
C_mean = np.mean(cs, axis=0)
C_inv = np.linalg.inv(C_mean)
w = mean_response @ C_inv
K = w @ mean_response.T
G = np.zeros_like(K)
for s_i in responses[1]:
    G += w @ (s_i - mean_response).T
G /= n_trials

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
sep = np.prod([separability(r0, r1, metric="cosine") for r0, r1 in zip(responses[0], responses[1])], axis=0)

# fit dual-exponential to envelope of separability and kernel diagonal
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
bounds = ([0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0], [1.0, 100.0, 1.0, 1.0, 1e2, 1e3, 5e1])
ir_params, ir_fit, sep = impulse_response_fit(sep, time, f=dualexponential, bounds=bounds, p0=p0)
kernel_params, kernel_fit, K_diag = impulse_response_fit(K_diag, time, f=dualexponential, bounds=bounds, p0=p0)

# calculate dimensionality in the impulse response period
ir_window = int(2*ir_params[-2]/dts)
ir = impulse_responses[1]["mean"][:ir_window, :]
ir_reduced = ir[:, np.mean(ir, axis=0) > epsilon]
n_neurons = ir_reduced.shape[1]
dim_sep = get_dim(ir, center=False) / N_e
dim_sep_r = get_dim(ir_reduced, center=False) / n_neurons
dim_sep_c = get_dim(ir, center=True) / N_e
dim_sep_rc = get_dim(ir_reduced, center=True) / n_neurons
neuron_dropout = (N_e - n_neurons) / N_e

# save results
results = {"g": g, "Delta": Delta, "ei_ratio": ei_ratio,
           "s_mean": s_mean, "s_std": s_std, "ff_between": ffs, "ff_within": ffs2, "ff_windows": taus,
           "dim_ss": dim_ss, "dim_ss_r": dim_ss_r, "dim_ss_c": dim_ss_c, "dim_ss_rc": dim_ss_rc,
           "dim_ir": dim_irs[0], "dim_ir_r": dim_irs[1], "dim_ir_c": dim_irs[2], "dim_ir_rc": dim_irs[3],
           "dim_sep": dim_sep, "dim_sep_r": dim_sep_r, "dim_sep_c": dim_sep_c, "dim_sep_rc": dim_sep_rc,
           "sep_ir": sep, "fit_ir": ir_fit, "params_ir": ir_params,
           "fit_kernel": kernel_fit, "params_kernel": kernel_params,
           "neuron_dropout": neuron_dropout, "impulse_responses": impulse_responses,
           "K_mean": K_mean, "K_var": K_var, "K_diag": K_diag, "G_sum": kernel_var
           }
pickle.dump(results, open(f"{path}/dim_eir{int(10*ei_ratio)}_g{int(10*g)}_D{int(Delta)}_{rep+1}.pkl", "wb"))

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
# fig.suptitle(f"D = {dim_ss}, D_r = {dim_ss_r}, D_c = {dim_ss_c}, D_rc = {dim_ss_rc}")
# plt.tight_layout()
#
# # impulse response dynamics
# fig, axes = plt.subplots(nrows=2, figsize=(12, 5))
# ax = axes[0]
# for c in impulse_responses:
#     ax.plot(time, np.mean(impulse_responses[c]["mean"], axis=1), label=f"input {c}")
# ax.set_xlabel("time (ms)")
# ax.set_ylabel("r (Hz)")
# ax.set_title("Mean-field response")
# ax.legend()
# ax = axes[1]
# ax.plot(time, sep, label="target IR")
# ax.plot(time, ir_fit, label="fitted IR")
# ax.legend()
# ax.set_xlabel("time (ms)")
# ax.set_ylabel("S")
# ax.set_title("Input Separability")
# fig.suptitle(f"D = {dim_sep}, D_rc = {dim_sep_rc}, tau_s = {ir_params[-2]}, tau_f = {ir_params[-1]}")
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
# fig.suptitle(f"D = {dim_irs[0]}, D_r = {dim_irs[1]}, D_c = {dim_irs[2]}, D_rc = {dim_irs[3]}")
# plt.tight_layout()
#
# # kernel statistics
# fig, axes = plt.subplots(nrows=3, figsize=(12, 9))
# ax = axes[0]
# ax.plot(time, K_mean)
# ax.set_ylabel("mean(K)")
# ax = axes[1]
# ax.plot(time, K_var)
# ax.set_ylabel("var(K)")
# ax = axes[2]
# ax.plot(time, K_diag, label="target")
# ax.plot(time, kernel_fit, label="fit")
# ax.set_xlabel("time (ms)")
# ax.set_ylabel("diag(K)")
# ax.set_title(f"tau_s = {kernel_params[-2]}, tau_f = {kernel_params[-1]}")
# fig.suptitle("Kernel statistics")
# plt.tight_layout()
#
# # plotting fano factor distributions at different time scales
# fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
# ax1, ax2 = axes
# for tau, ff1, ff2 in zip(taus, ffs, ffs2):
#     ax1.hist(ff1, label=f"tau = {tau}", alpha=0.5)
#     ax1.set_xlabel("ff")
#     ax1.set_ylabel("#")
#     ax2.hist(ff2, label=f"tau = {tau}", alpha=0.5)
#     ax2.set_xlabel("ff")
#     ax2.set_ylabel("#")
# ax1.set_title("time-specific FFs")
# ax1.legend()
# ax2.set_title("neuron-specific FFs")
# ax2.legend()
# fig.suptitle("Fano Factor Distributions")
# plt.tight_layout()
#
# plt.show()
