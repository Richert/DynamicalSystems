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

# general model parameters
N = 200
E_e = 0.0
E_i = -65.0
v_spike = 50.0
v_reset = -90.0
g_in = 10.0

# get sweep condition
rep = 0 #int(sys.argv[-1])
g = 10.0 #float(sys.argv[-2])
Delta = 4.0 #float(sys.argv[-3])
p = 0.4 #float(sys.argv[-4])
path = "" #str(sys.argv[-5])

# input parameters
dt = 1e-2
dts = 1e-1
p_in = 0.6
dur = 20.0
window = 1000.0
n_trials = 5
amp = 1e-2
cutoff = 1000.0

# exc parameters
C = 20.0
k = 1.0
v_r = -65.0
v_t = -40.0
eta = 0.0
a = 0.2
b = 2.0
d = 0.0
s_e = 45.0*1e-3
tau_s = 6.0

# connectivity parameters
g_norm = g / np.sqrt(N * p)
W = random_connectivity(N, N, p, normalize=False)

# define distribution of etas
f = gaussian if theta_dist == "gaussian" else lorentzian
thetas = f(N, loc=v_t, scale=Delta, lb=v_r, ub=2 * v_t - v_r)

# initialize the model
######################

# initialize operators
op = OperatorTemplate.from_yaml("config/ik_snn/ik_op")
exc_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": thetas, "eta": eta, "tau_u": 1 / a, "b": b, "kappa": d,
            "g_e": 0.0, "E_i": E_i, "tau_s": tau_s, "v": v_t, "g_i": g_norm, "E_e": E_e}

# initialize model
net = Network(dt, device=device)
net.add_diffeq_node("ik", f"config/ik_snn/ik", weights=W, source_var="s", target_var="s_i",
                    input_var="g_e_in", output_var="s", spike_var="spike", reset_var="v", to_file=False,
                    node_vars=exc_vars.copy(), op="ik_op", spike_reset=v_reset, spike_threshold=v_spike,
                    clear=True)

# simulation 1: steady-state
############################

# define input
T = cutoff + window
inp = np.zeros((int(T/dt), N))
inp[:, :] += poisson.rvs(mu=s_e*g_in*dt, size=(int(T/dt), N))
inp = convolve_exp(inp, tau_s, dt)

# perform cutoff simulation
_ = net.run(inputs=inp[:int(cutoff/dt), :], sampling_steps=int(dts/dt), record_output=False, verbose=False,
            enable_grad=False)

# perform steady-state simulation
obs = net.run(inputs=inp[int(cutoff/dt):, :], sampling_steps=int(dts/dt), record_output=True, verbose=False,
              enable_grad=False)
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
in_split = int(0.5*len(inp_neurons))
dur_tmp = int(dur/dt)

# collect network responses to stimulation
ir1s, ir2s, ir0s = [], [], []
for trial in range(n_trials):

    noise = poisson.rvs(mu=s_e*g_in*dt, size=(int(window/dt), N))

    # no input
    ##########

    # generate random input for first trial
    inp = np.zeros((int(window / dt), N))
    inp[:, :] += noise

    # get network response to input
    obs = net.run(inputs=convolve_exp(inp, tau_s, dt), sampling_steps=int(dts / dt), record_output=True,
                  verbose=False, enable_grad=False)
    ir0 = obs.to_numpy("out") * 1e3 / tau_s
    ir0s.append(ir0)

    # input 1
    #########

    # generate random input for first trial
    inp = np.zeros((int(window / dt), N))
    inp[:, :] += noise
    inp[:dur_tmp, inp_neurons[:in_split]] += poisson.rvs(mu=amp * g_in * dt, size=(dur_tmp, in_split))

    # get network response to input
    obs = net.run(inputs=convolve_exp(inp, tau_s, dt), sampling_steps=int(dts / dt), record_output=True,
                  verbose=False, enable_grad=False)
    ir1 = obs.to_numpy("out") * 1e3 / tau_s
    ir1s.append(ir1)

    # input 2
    #########

    # generate random input for first trial
    inp = np.zeros((int(window / dt), N))
    inp[:, :] += noise
    inp[:dur_tmp, inp_neurons[in_split:]] += poisson.rvs(mu=amp * g_in * dt, size=(dur_tmp, in_split))

    # get network response to input
    obs = net.run(inputs=convolve_exp(inp, tau_s, dt), sampling_steps=int(dts / dt), record_output=True,
                  verbose=False, enable_grad=False)
    ir2 = obs.to_numpy("out") * 1e3 / tau_s
    ir2s.append(ir2)

    print(f"Finished {trial + 1} of {n_trials} trials.")

# calculate trial-averaged network response
ir0 = np.asarray(ir0s)
ir1 = np.asarray(ir1s)
ir2 = np.asarray(ir2s)
ir_mean0 = np.mean(ir0, axis=0)
ir_mean1 = np.mean(ir1, axis=0)
ir_mean2 = np.mean(ir2, axis=0)
ir_std1 = np.std(ir1, axis=0)
ir_std2 = np.std(ir2, axis=0)

# calculate separability
sep_12 = separability(ir_mean1, ir_mean2, metric="cosine")
sep_01 = separability(ir_mean1, ir_mean0, metric="cosine")
sep_02 = separability(ir_mean2, ir_mean0, metric="cosine")
sep = sep_12*sep_01*sep_02

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

# fit impulse response of mean-field
ir0 = np.mean(ir_mean0, axis=1)
ir1 = np.mean(ir_mean1, axis=1)
ir2 = np.mean(ir_mean2, axis=1)
diff = (ir0 - ir1)**2
params_mf, ir_mf = impulse_response_fit(diff, time, f=dualexponential, bounds=bounds, p0=p0)

# calculate dimensionality in the impulse response period
ir_window = int(20.0*params[-2])
dim_ir1 = get_dim(ir_mean1[:ir_window, :], center=True)
dim_ir2 = get_dim(ir_mean2[:ir_window, :], center=True)
dim_ir = (dim_ir1 + dim_ir2)/2
ir_mean1_tmp = ir_mean1[:ir_window, np.sum(ir_mean1[:ir_window, :], axis=0) > 0]
ir_mean2_tmp = ir_mean2[:ir_window, np.sum(ir_mean1[:ir_window, :], axis=0) > 0]
dim_ir1 = get_dim(ir_mean1_tmp, center=True)
dim_ir2 = get_dim(ir_mean2_tmp, center=True)
dim_ir_r = (dim_ir1 + dim_ir2)/2
N_ir1 = ir_mean1_tmp.shape[1]
N_ir2 = ir_mean2_tmp.shape[1]
dim_ir1 = get_dim(ir_mean1[:ir_window, :], center=False)
dim_ir2 = get_dim(ir_mean2[:ir_window, :], center=False)
dim_ir_nc = (dim_ir1 + dim_ir2)/2

# save results
results = {"g": g, "Delta": Delta, "p": p,
           "dim_ss": dim_ss, "s_mean": s_mean, "s_std": s_std, "ff_between": ffs, "ff_within": ffs2, "ff_windows": taus,
           "dim_ir": dim_ir, "sep_ir": sep, "fit_ir": ir_fit, "params_ir": params, "mean_ir0": ir0,
           "mean_ir1": ir1, "std_ir1": np.mean(ir_std1, axis=1),
           "mean_ir2": ir2, "std_ir2": np.mean(ir_std2, axis=1),
           "mf_params_ir": params_mf, "dim_ss_reduced": dim_ss_r, "dim_ir_reduced": dim_ir_r,
           "dim_ss_nc": dim_ss_nc, "dim_ir_nc": dim_ir_nc, "N": N, "N_ss": N_ss, "N_ir1": N_ir1, "N_ir2": N_ir2
           }
# pickle.dump(results, open(f"{path}/dim2_inh_g{int(10*g)}_D{int(Delta)}_p{int(10*p)}_{rep+1}.pkl", "wb"))

# plotting firing rate dynamics
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(s_mean*1e3, label="mean(r)")
ax.plot(s_std*1e3, label="std(r)")
ax.legend()
ax.set_xlabel("steps")
ax.set_ylabel("r")
ax.set_title(f"Dim = {dim_ss}")
fig.suptitle("Mean-field rate dynamics")
plt.tight_layout()

# plotting spikes
fig, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(s.T, aspect="auto", interpolation="none", cmap="Greys")
plt.colorbar(im, ax=ax)
ax.set_xlabel("steps")
ax.set_ylabel("neurons")
fig.suptitle("Spiking dynamics")
plt.tight_layout()

# plotting impulse response
fig, axes = plt.subplots(nrows=2, figsize=(12, 4))
fig.suptitle("Impulse response")
ax = axes[0]
ax.plot(ir0, label="no input")
ax.plot(ir1, label="input")
ax.plot(diff, label="diff.")
ax.plot(ir_mf, label="diff. fit")
ax.set_xlabel("steps")
ax.set_ylabel("r (Hz)")
ax.legend()
ax.set_title(f"Impulse Response")
ax = axes[1]
ax.plot(sep, label="combined IR")
ax.plot(ir_fit, label="exp. fit")
ax.legend()
ax.set_xlabel("steps")
ax.set_ylabel("SR")
ax.set_title(f"Dim = {np.round(results['dim_ir'], decimals=1)}, tau = {np.round(params[-1], decimals=1)}, "
             f"tau_mf = {np.round(params_mf[-1], decimals=1)}")

plt.tight_layout()
plt.show()
