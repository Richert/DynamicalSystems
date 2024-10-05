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

# meta parameters
device = "cpu"
theta_dist = "gaussian"

# general model parameters
N = 1000
E_e = 0.0
E_i = -65.0
v_spike = 50.0
v_reset = -90.0
g_in = 10.0

# get sweep condition
rep = 0 #int(sys.argv[-1])
g = 2.0 #float(sys.argv[-2])
Delta_e = 0.1 #float(sys.argv[-3])
Delta_i = 0.2 #float(sys.argv[-4])
path = "" #str(sys.argv[-5])

# input parameters
dt = 1e-2
dts = 1e-1
p_in = 0.6
dur = 20.0
window = 1000.0
n_trials = 10
amplitudes = np.linspace(1e-4, 1e-2, num=3)
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

# perform cutoff simulation
_ = net.run(inputs=inp[:int(cutoff/dt), :], sampling_steps=int(dts/dt), record_output=False, verbose=False,
            enable_grad=False)
y0 = {key: val[:] for key, val in net.state.items()}

# perform steady-state simulation
obs = net.run(inputs=inp[int(cutoff/dt):, :], sampling_steps=int(dts/dt), record_output=True, verbose=False,
              enable_grad=False)
s = obs.to_dataframe("out")
s.iloc[:, :N_e] /= tau_s_e
s.iloc[:, N_e:] /= tau_s_i

# calculate dimensionality in the steady-state period
s_vals = s.values[:, :N_e]
dim_ss = get_dim(s_vals)

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
results = {"g": g, "Delta_e": Delta_e, "Delta_i": Delta_i,
           "dim_ss": dim_ss, "s_mean": s_mean, "s_std": s_std, "ff_between": ffs, "ff_within": ffs2, "ff_windows": taus,
           "amp_ir": [], "dim_ir1": [], "dim_ir2": [], "sep_ir": [], "mean_ir1": [], "std_ir1": [],
           "mean_ir2": [], "std_ir2": []
           }
for amp in amplitudes:

    d1s, d2s, ir1s, ir2s = [], [], [], []

    for trial in range(n_trials):

        # input 1
        #########

        # generate random input for first trial
        inp = np.zeros((int(window / dt), N))
        inp[:, :N_e] += poisson.rvs(mu=s_e * g_in * dt, size=(inp.shape[0], N_e))
        inp[:, N_e:] += poisson.rvs(mu=s_i * g_in * dt, size=(inp.shape[0], N_i))
        inp[:dur_tmp, inp_neurons[:in_split]] += poisson.rvs(mu=amp * g_in * dt, size=(dur_tmp, in_split))

        # get network response to input
        obs = net.run(inputs=convolve_exp(inp, tau_s_e, dt), sampling_steps=int(dts / dt), record_output=True,
                      verbose=False, enable_grad=False)
        ir1 = obs.to_numpy("out") * 1e3/tau_s_e
        ir1s.append(ir1)

        # calculate dimensionality
        d1s.append(get_dim(ir1[:, :N_e]))

        # input 2
        #########

        # set initial state
        net.reset(state=y0)

        # generate random input for first trial
        inp = np.zeros((int(window / dt), N))
        inp[:, :N_e] += poisson.rvs(mu=s_e * g_in * dt, size=(inp.shape[0], N_e))
        inp[:, N_e:] += poisson.rvs(mu=s_i * g_in * dt, size=(inp.shape[0], N_i))
        inp[:dur_tmp, inp_neurons[in_split:]] += poisson.rvs(mu=amp * g_in * dt, size=(dur_tmp, in_split))

        # get network response to input
        obs = net.run(inputs=convolve_exp(inp, tau_s_e, dt), sampling_steps=int(dts / dt), record_output=True,
                      verbose=False, enable_grad=False)
        ir2 = obs.to_numpy("out") * 1e3/tau_s_e
        ir2s.append(ir2)

        # calculate dimensionality
        d2 = get_dim(ir2[:, :N_e])
        d2s.append(d2)

        # test plotting
        ###############

        # fig, ax = plt.subplots(figsize=(12, 4))
        # ax.plot(np.mean(ir1, axis=1), label="IR1")
        # ax.plot(np.mean(ir2, axis=1), label="IR2")
        # ax.set_xlabel("steps")
        # ax.set_ylabel("r")
        # ax.legend()
        # plt.tight_layout()
        # plt.show()

    # calculate trial-averaged network response
    ir1 = np.asarray(ir1s)
    ir2 = np.asarray(ir2s)
    ir_mean1 = np.mean(ir1, axis=0)
    ir_std1 = np.std(ir1, axis=0)
    ir_mean2 = np.mean(ir2, axis=0)
    ir_std2 = np.std(ir2, axis=0)

    # calculate separability
    sep = separability(ir_mean1, ir_mean2, metric="correlation")

    # save trial-averaged results
    results["amp_ir"].append(amp)
    results["dim_ir1"].append(np.mean(d1s))
    results["dim_ir2"].append(np.mean(d2s))
    results["mean_ir1"].append(np.mean(ir_mean1, axis=1))
    results["std_ir1"].append(np.mean(ir_std1, axis=1))
    results["mean_ir2"].append(np.mean(ir_mean2, axis=1))
    results["std_ir2"].append(np.mean(ir_std2, axis=1))
    results["sep_ir"].append(sep)

# save results
# pickle.dump({"g": g, "Delta_e": Delta_e, "Delta_i": Delta_i, "dim_ss": dim_ss, "dim_ir": dim_ir,
#              "s_mean": s_mean, "s_std": s_std, "ff_between": ffs, "ff_within": ffs2, "ff_windows": taus,
#              "ir_target": ir, "ir_fit": ir_fit, "ir_params": p},
#             open(f"{path}/eic_g{int(g)}_De{int(Delta_e)}_Di{int(Delta_i)}_{rep+1}.pkl", "wb"))

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
fig, axes = plt.subplots(nrows=len(amplitudes), ncols=2, figsize=(12, 2*len(amplitudes)))
fig.suptitle("Impulse response")
for i, amp in enumerate(amplitudes):
    ax = axes[i, 0]
    ir_mean = results["mean_ir2"][i] - s_mean
    ax.plot(ir_mean, label="mean")
    ax.plot(results["std_ir2"][i] / ir_mean, label="std")
    ax.set_xlabel("steps")
    ax.set_ylabel("r (Hz)")
    ax.legend()
    ax.set_title(f"Avg. impulse amplitude = {amp} pA")
    ax = axes[i, 1]
    ax.plot(results["sep_ir"][i])
    ax.set_xlabel("steps")
    ax.set_ylabel("separability")
    ax.set_title(f"Dim = {np.round(results['dim_ir1'][i], decimals=1)}")

plt.tight_layout()
plt.show()
