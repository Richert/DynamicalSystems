from rectipy import Network, random_connectivity
from pyrates import CircuitTemplate, NodeTemplate, OperatorTemplate
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import poisson, expon
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
g = 1.0 #float(sys.argv[-2])
Delta_e = 1.0 #float(sys.argv[-3])
Delta_i = 1.0 #float(sys.argv[-4])
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
s_e = 15.0*1e-3
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
s_i = 15.0*1e-3
tau_s_i = 10.0

# connectivity parameters
p_ee = 0.1
p_ii = 0.4
p_ie = 0.2
p_ei = 0.4
g_ee = g / np.sqrt(N_e * p_ee)
g_ii = g / np.sqrt(N_i * p_ii)
g_ei = g / np.sqrt(N_i * p_ei)
g_ie = g / np.sqrt(N_e * p_ie)
W_ee = random_connectivity(N_e, N_e, p_ee, normalize=False)
W_ie = random_connectivity(N_i, N_e, p_ie, normalize=False)
W_ei = random_connectivity(N_e, N_i, p_ei, normalize=False)
W_ii = random_connectivity(N_i, N_i, p_ii, normalize=False)

# define distribution of etas
f = gaussian if theta_dist == "gaussian" else lorentzian
thetas_e = f(N_e, mu=v_t_e, delta=Delta_e, lb=v_r_e, ub=2*v_t_e-v_r_e)
thetas_i = f(N_i, mu=v_t_i, delta=Delta_i, lb=v_r_i, ub=2*v_t_i-v_r_i)

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
T = cutoff + window
inp = np.zeros((int(T/dt), N))
inp[:, :N_e] += poisson.rvs(mu=s_e*g_in*dt, size=(inp.shape[0], N_e))
inp[:, N_e:] += poisson.rvs(mu=s_i*g_in*dt, size=(inp.shape[0], N_i))
inp = convolve_exp(inp, tau_s_e, dt)

# perform simulation
obs = net.run(inputs=inp, sampling_steps=int(dts/dt), record_output=True, verbose=False, enable_grad=False,
              cutoff=int(cutoff/dt))
s = obs.to_dataframe("out")
s.iloc[:, :N_e] /= tau_s_e
s.iloc[:, N_e:] /= tau_s_i

# calculate dimensionality in the steady-state period
dim_ss = get_dim(s.values[:, :N_e])

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

# preparations
inp_neurons = np.random.choice(N_e, size=(int(N_e*p_in),))
in_split = int(0.5*len(inp_neurons))
dur_tmp = int(dur/dt)

# collect network responses to stimulation
ir1s, ir2s, ir0s = [], [], []
for trial in range(n_trials):

    noise_e = poisson.rvs(mu=s_e*g_in*dt, size=(int(window/dt), N_e))
    noise_i = poisson.rvs(mu=s_i*g_in*dt, size=(int(window/dt), N_i))
    y0 = {key: val[:] for key, val in net.state.items()}

    # no input
    ##########

    # set initial state
    # net.reset(state=y0)

    # generate random input
    inp = np.zeros((int(window / dt), N))
    inp[:, :N_e] += noise_e
    inp[:, N_e:] += noise_i

    # get network response to input
    obs = net.run(inputs=convolve_exp(inp, tau_s_e, dt), sampling_steps=int(dts / dt), record_output=True,
                  verbose=False, enable_grad=False)
    ir0 = obs.to_numpy("out")[:, :N_e] * 1e3 / tau_s_e
    ir0s.append(ir0)

    # input 1
    #########

    # set initial state
    # net.reset(state=y0)

    # generate random input
    inp = np.zeros((int(window / dt), N))
    inp[:, :N_e] += noise_e
    inp[:, N_e:] += noise_i
    inp[:dur_tmp, inp_neurons[:in_split]] += poisson.rvs(mu=amp*g_in*dt, size=(dur_tmp, in_split))

    # get network response to input
    obs = net.run(inputs=convolve_exp(inp, tau_s_e, dt), sampling_steps=int(dts / dt), record_output=True,
                  verbose=False, enable_grad=False)
    ir1 = obs.to_numpy("out")[:, :N_e] * 1e3 / tau_s_e
    ir1s.append(ir1)

    # input 2
    #########

    # set initial state
    # net.reset(state=y0)

    ## generate random input
    inp = np.zeros((int(window / dt), N))
    inp[:, :N_e] += noise_e
    inp[:, N_e:] += noise_i
    inp[:dur_tmp, inp_neurons[in_split:]] += poisson.rvs(mu=amp*g_in*dt, size=(dur_tmp, in_split))

    # get network response to input
    obs = net.run(inputs=convolve_exp(inp, tau_s_e, dt), sampling_steps=int(dts / dt), record_output=True,
                  verbose=False, enable_grad=False)
    ir2 = obs.to_numpy("out")[:, :N_e] * 1e3 / tau_s_e
    ir2s.append(ir2)

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

# fit bi-exponential to envelope of impulse response
tau = 10.0
scale = 0.5
delay = 5.0
offset = 0.1
p0 = [offset, delay, scale, tau]
time = s.index.values
time = time - np.min(time)
bounds = ([0.0, 1.0, 0.0, 1.0], [1.0, 100.0, 1.0, 2e2])
params, ir_fit = impulse_response_fit(sep, time, f=alpha, bounds=bounds, p0=p0)

# fit impulse response of mean-field
ir0 = np.mean(ir_mean0, axis=1)
ir1 = np.mean(ir_mean1, axis=1)
ir2 = np.mean(ir_mean2, axis=1)
diff = (ir0 - ir1)**2
params_mf, ir_mf = impulse_response_fit(diff, time, f=alpha, bounds=bounds, p0=p0)

# calculate dimensionality in the impulse response period
ir_window = int(1e2*params[-1])
dim_ir1 = get_dim(ir_mean1[:ir_window, :])
dim_ir2 = get_dim(ir_mean2[:ir_window, :])
dim_ir = (dim_ir1 + dim_ir2)/2

# save results
results = {"g": g, "Delta_e": Delta_e, "Delta_i": Delta_i,
           "dim_ss": dim_ss, "s_mean": s_mean, "s_std": s_std, "ff_between": ffs, "ff_within": ffs2, "ff_windows": taus,
           "dim_ir": dim_ir, "sep_ir": sep, "fit_ir": ir_fit, "params_ir": params, "mean_ir0": ir0,
           "mean_ir1": ir1, "std_ir1": np.mean(ir_std1, axis=1),
           "mean_ir2": ir2, "std_ir2": np.mean(ir_std2, axis=1),
           "mf_params_ir": params_mf
           }
# pickle.dump(results, open(f"{path}/dim_eic_g{int(g)}_De{int(Delta_e)}_Di{int(Delta_i)}_{rep+1}.pkl", "wb"))

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
ax.plot(results["mean_ir0"], label="no input")
ax.plot(results["mean_ir1"], label="input 1")
ax.plot(results["mean_ir2"], label="input 2")
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
ax.set_title(f"Dim = {np.round(results['dim_ir'], decimals=1)}, tau = {np.round(params[-1], decimals=1)}")

plt.tight_layout()
plt.show()
