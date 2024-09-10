from rectipy import Network, random_connectivity
from pyrates import CircuitTemplate, NodeTemplate, OperatorTemplate
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import poisson
from scipy.signal import find_peaks
import sys
from custom_functions import *


# define parameters
###################

# get sweep condition
rep = 0 #int(sys.argv[-1])
g = 4.0 #float(sys.argv[-2])
Delta = 3.0 #float(sys.argv[-3])

# general parameters
E_e = 0.0
E_i = -65.0
v_spike = 50.0
v_reset = -90.0
theta_dist = "gaussian"
N = 100
g_in = 1

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
d_e = 40.0
s_e = 10.0*1e-3
tau_s_e = 6.0

# inh parameters
p_i = 1-p_e
N_i = N-N_e
C_i = 100.0
k_i = 0.7
v_r_i = -75.0
v_t_i = -55.0
eta_i = 0.0
a_i = 0.03
b_i = -2.0
d_i = 100.0
s_i = 10.0*1e-3
tau_s_i = 10.0

# connectivity parameters
p_ee = 0.2 # 0.2
p_ii = 0.3 # 0.3
p_ie = 0.3 # 0.3
p_ei = 0.2 # 0.2

# define distribution of etas
f = gaussian if theta_dist == "gaussian" else lorentzian
thetas_e = f(N_e, mu=v_t_e, delta=Delta, lb=v_r_e, ub=2*v_t_e-v_r_e)
thetas_i = f(N_i, mu=v_t_i, delta=Delta, lb=v_r_i, ub=2*v_t_i-v_r_i)

# random connectivity
W_ee = random_connectivity(N_e, N_e, p_ee, normalize=False)
W_ie = random_connectivity(N_i, N_e, p_ie, normalize=False)
W_ei = random_connectivity(N_e, N_i, p_ei, normalize=False)
W_ii = random_connectivity(N_i, N_i, p_ii, normalize=False)

# define inputs
T = 2500.0
cutoff = 1000.0
start = 1000.0
stop = 1020.0
amp = 200.0*1e-3
dt = 1e-2
dts = 1e-1
inp = np.zeros((int(T/dt), N))
inp[:, :N_e] += poisson.rvs(mu=s_e*g_in*dt, size=(inp.shape[0], N_e))
inp[:, N_e:] += poisson.rvs(mu=s_i*g_in*dt, size=(inp.shape[0], N_i))
inp[int((cutoff+start)/dt):int((cutoff+stop)/dt), :N_e] += poisson.rvs(mu=amp*g_in*dt, size=(int((stop-start)/dt), N_e))
inp = convolve_exp(inp, tau_s_e, dt)

# run the model
###############

# initialize operators
op = OperatorTemplate.from_yaml("config/ik_snn/ik_op")
exc_vars = {"C": C_e, "k": k_e, "v_r": v_r_e, "v_theta": thetas_e, "eta": eta_e, "tau_u": 1/a_e, "b": b_e, "kappa": d_e,
            "g_e": g, "E_i": E_i, "tau_s": tau_s_e, "v": v_t_e, "g_i": g, "E_e": E_e}
inh_vars = {"C": C_i, "k": k_i, "v_r": v_r_i, "v_theta": thetas_i, "eta": eta_i, "tau_u": 1/a_i, "b": b_i, "kappa": d_i,
            "g_e": g, "E_i": E_i, "tau_s": tau_s_i, "v": v_t_i, "g_i": g, "E_e": E_e}

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
net = Network(dt, device="cpu")
net.add_diffeq_node("eic", eic, input_var="g_e_in", output_var="s", spike_var="spike", reset_var="v",
                    to_file=False, op="ik_op", spike_reset=v_reset, spike_threshold=v_spike, clear=True, N=N,
                    record_vars=["v"])

# perform simulation
obs = net.run(inputs=inp, sampling_steps=int(dts/dt), record_output=True, verbose=False, enable_grad=False,
              cutoff=int(cutoff/dt), record_vars=[("eic", "v", False)])
s = obs.to_dataframe("out")
v = obs.to_numpy(("eic", "v"))
s.iloc[:, :N_e] /= tau_s_e
s.iloc[:, N_e:] /= tau_s_i

# calculate dimensionality in the steady-state period
idx_stop = int(start/dts)
s_vals = s.values[:idx_stop, :N_e]
dim_ss = get_dim(s_vals)

# calculate dimensionality in the impulse response period
s_vals = s.values[idx_stop:, :N_e]
dim_ir = get_dim(s_vals)

# extract spikes in network
spike_counts = []
s_vals = s.values[:idx_stop, :]
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

# fit bi-exponential to envelope of impulse response
tau = 10.0
a = 10.0
d = 3.0
p0 = [d, a, tau]
ir = s_mean[int(start/dts):] * 1e3
ir = ir - np.min(ir)
time = s.index.values[int(start/dts):]
time = time - np.min(time)
bounds = ([0.0, 1.0, 1e-1], [1e2, 3e2, 5e2])
p, ir_fit = impulse_response_fit(ir, time, f=alpha, p0=p0, bounds=bounds, gtol=None, loss="linear")

# save results
# pickle.dump({"g": g, "Delta": Delta, "theta_dist": theta_dist, "dim": dim, "s_mean": s_mean, "s_std": s_std},
#             open(f"/media/fsmresfiles/richard_data/numerics/dimensionality/bal_ss_g{int(g)}_D{int(Delta*10)}_{rep+1}.p",
#                  "wb"))

# plotting firing rate dynamics
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(s_mean*1e3, label="mean(r)")
ax.plot(s_std*1e3, label="std(r)")
ax.axvline(x=int(start/dts), linestyle="dashed", color="black")
ax.axvline(x=int(stop/dts), linestyle="dashed", color="black")
ax.legend()
ax.set_xlabel("steps")
ax.set_ylabel("r")
ax.set_title(f"Dim = {dim_ss}")
fig.suptitle("Mean-field rate dynamics")
plt.tight_layout()

# plotting impulse response
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(ir, label="Target IR")
ax.plot(ir_fit, label="Fitted IR")
ax.legend()
ax.set_xlabel("steps")
ax.set_ylabel("r (Hz)")
ax.set_title(f"Dim = {dim_ir}, time constant: tau = {p[2]} ms")
fig.suptitle("Mean-field impulse response")
plt.tight_layout()

# plotting spikes
fig, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(s.T, aspect="auto", interpolation="none", cmap="Greys")
plt.colorbar(im, ax=ax)
ax.set_xlabel("steps")
ax.set_ylabel("neurons")
fig.suptitle("Spiking dynamics")
plt.tight_layout()

# plotting membrane potential dynamics
_, axes = plt.subplots(nrows=2, figsize=(12, 8))
ax = axes[0]
ax.plot(np.mean(v, axis=1))
ax.set_ylabel("v (mV)")
ax.set_title("Mean-field membrane potential dynamics")
ax = axes[1]
for neuron_idx in np.random.choice(N, replace=False, size=(5,)):
    ax.plot(v[:, neuron_idx], label=f"neuron {neuron_idx}")
ax.legend()
ax.set_xlabel("steps")
ax.set_ylabel("v (mV)")
ax.set_title("Single neuron traces")
plt.tight_layout()

# plotting input
_, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(inp.T, aspect="auto")
plt.colorbar(im, ax=ax)
ax.set_xlabel("time")
ax.set_ylabel("neurons")
ax.set_title("Extrinsic input")
plt.tight_layout()
plt.show()
