from rectipy import FeedbackNetwork, circular_connectivity, random_connectivity
from pyrates import CircuitTemplate, NodeTemplate
import sys
sys.path.append("~/PycharmProjects/DynamicalSystems/reservoir_computing")
import numpy as np
from scipy.stats import cauchy, rv_discrete
import matplotlib.pyplot as plt
from matplotlib import colors
import pickle
from scipy.signal import find_peaks


def lorentzian(n: int, eta: float, delta: float, lb: float, ub: float):
    samples = np.zeros((n,))
    for i in range(n):
        s = cauchy.rvs(loc=eta, scale=delta)
        while s <= lb or s >= ub:
            s = cauchy.rvs(loc=eta, scale=delta)
        samples[i] = s
    return samples


def dist(x: int, method: str = "inverse", zero_val: float = 1.0, inverse_pow: float = 1.0) -> float:
    if method == "inverse":
        return 1/x**inverse_pow if x > 0 else zero_val
    if method == "exp":
        return np.exp(-x) if x > 0 else zero_val
    else:
        raise ValueError("Invalid method.")


# define parameters
###################

# general parameters
N = 1000
p = 0.5
v_spike = 1e3
v_reset = -1e3

# RS neuron parameters
Ce = 100.0   # unit: pF
ke = 0.7  # unit: None
ve_r = -60.0  # unit: mV
ve_t = -40.0  # unit: mV
Delta_e = 0.5  # unit: mV
de = 10.0
ae = 0.03
be = -2.0
Ie = 60.0

# FS neuron parameters
Ci = 20.0   # unit: pF
ki = 1.0  # unit: None
vi_r = -55.0  # unit: mV
vi_t = -40.0  # unit: mV
Delta_i = 2.0  # unit: mV
di = 0.0
ai = 0.2
bi = 0.025
Ii = 35.0

# synaptic parameters
g_ampa = 1.0
g_gaba = 1.0
E_ampa = 0.0
E_gaba = -65.0
tau_ampa = 6.0
tau_gaba = 8.0
k_ee = 16.0
k_ei = 16.0
k_ie = 4.0
k_ii = 4.0

# define lorentzian of etas
spike_thresholds_e = lorentzian(N, eta=ve_t, delta=Delta_e, lb=ve_r, ub=2*ve_t - ve_r)
spike_thresholds_i = lorentzian(N, eta=vi_t, delta=Delta_i, lb=vi_r, ub=2*vi_t - vi_r)

# define connectivity
# indices = np.arange(0, N, dtype=np.int32)
# e_pdfs = np.asarray([dist(idx, method="inverse", zero_val=0.0, inverse_pow=1.5) for idx in indices])
# e_pdfs /= np.sum(e_pdfs)
# i_pdfs = np.asarray([dist(idx, method="inverse", zero_val=0.0, inverse_pow=0.0) for idx in indices])
# i_pdfs /= np.sum(i_pdfs)
# W_ee = circular_connectivity(N, p, spatial_distribution=rv_discrete(values=(indices, e_pdfs)), homogeneous_weights=False)
# W_ie = circular_connectivity(N, p, spatial_distribution=rv_discrete(values=(indices, i_pdfs)), homogeneous_weights=False)
# W_ei = circular_connectivity(N, p, spatial_distribution=rv_discrete(values=(indices, i_pdfs)), homogeneous_weights=False)
# W_ii = circular_connectivity(N, p, spatial_distribution=rv_discrete(values=(indices, i_pdfs)), homogeneous_weights=False)
W_ee = random_connectivity(N, N, p, normalize=True)
W_ie = random_connectivity(N, N, p, normalize=True)
W_ei = random_connectivity(N, N, p, normalize=True)
W_ii = random_connectivity(N, N, p, normalize=True)

# define inputs
T = 2500.0
cutoff = 500.0
dt = 1e-2
dts = 1e-1
inp = np.zeros((int(T/dt), N))
# p_in = 0.5
# n_inputs = int(p_in*N)
# center = int(N*0.5)
# inp_indices = np.arange(center-int(0.5*n_inputs), center+int(0.5*n_inputs))
# inp[:int(0.5*cutoff/dt), :N] -= 20.0
# inp[:int(0.5*cutoff/dt), N:] += 20.0
# inp[int(1000/dt):int(1500/dt), inp_indices] += 30.0
# inp[int(2500/dt):int(3000/dt), N + inp_indices] += 30.0
# inp[:int(0.5*cutoff/dt), :N] += 0.2
# inp[int(750.0/dt):int(2000/dt), N:] -= 0.2

# run the model
###############

# initialize nodes
e_vars = {"C": Ce, "k": ke, "v_r": ve_r, "v_theta": spike_thresholds_e, "eta": Ie, "tau_u": 1/ae, "b": be, "kappa": de,
          "g_e": g_ampa, "E_e": E_ampa, "g_i": g_gaba, "E_i": E_gaba, "tau_s": tau_ampa, "v": ve_t}
i_vars = {"C": Ci, "k": ki, "v_r": vi_r, "v_theta": spike_thresholds_i, "eta": Ii, "tau_u": 1/ai, "b": bi, "kappa": di,
          "g_e": g_ampa, "E_e": E_ampa, "g_i": g_gaba, "E_i": E_gaba, "tau_s": tau_gaba, "v": vi_t}
rs = NodeTemplate.from_yaml("config/ik_snn/ik")
fs = NodeTemplate.from_yaml("config/ik_snn/ik")

# construct rs and fs circuits
rs_neurons = {f'rs_{i}': rs for i in range(N)}
rs_net = CircuitTemplate("rs", nodes=rs_neurons)
rs_net.add_edges_from_matrix(source_var="ik_op/s", target_var="ik_op/s_e", weight=W_ee * k_ee,
                             source_nodes=list(rs_neurons.keys()))
rs_net.update_var(node_vars={f"all/ik_op/{key}": val for key, val in e_vars.items()})
fs_neurons = {f'fs_{i}': fs for i in range(N)}
fs_net = CircuitTemplate("fs", nodes=fs_neurons)
fs_net.add_edges_from_matrix(source_var="ik_op/s", target_var="ik_op/s_i", weight=W_ii * k_ii,
                             source_nodes=list(fs_neurons.keys()))
fs_net.update_var(node_vars={f"all/ik_op/{key}": val for key, val in i_vars.items()})

# combine the rs and fs populations into a single circuit
# net = CircuitTemplate("eic", circuits={"rs": rs_net, "fs": fs_net})
# net.add_edges_from_matrix(source_var="ik_op/s", target_var="ik_op/s_e", weight=W_ie * k_ie,
#                           source_nodes=[f"rs/rs_{i}" for i in range(N)], target_nodes=[f"fs/fs_{i}" for i in range(N)])
# net.add_edges_from_matrix(source_var="ik_op/s", target_var="ik_op/s_i", weight=W_ei * k_ei,
#                           source_nodes=[f"fs/fs_{i}" for i in range(N)], target_nodes=[f"rs/rs_{i}" for i in range(N)])

# initialize rectipy model
model = FeedbackNetwork(dt=dt, device="cuda:0")
model.add_diffeq_node("rs", node=rs_net, input_var="s_i", output_var="s",
                      spike_var="spike", spike_def="v", spike_reset=v_reset,
                      spike_threshold=v_spike, verbose=True, clear=True, op="ik_op")
model.add_diffeq_node("fs", node=fs_net, input_var="s_e", output_var="s",
                      spike_var="spike", spike_def="v", spike_reset=v_reset,
                      spike_threshold=v_spike, verbose=True, clear=True, op="ik_op")
model.add_edge("fs", "rs", weights=W_ei * k_ei, train=None, feedback=False)
model.add_edge("rs", "fs", weights=W_ie * k_ie, train=None, feedback=True)

# perform simulation
obs = model.run(inputs=inp, sampling_steps=int(dts/dt), record_output=False, verbose=True,
                record_vars=[("rs", "out", False), ("fs", "out", False)])
rs_res = obs.to_numpy(("rs", "out"))
fs_res = obs.to_numpy(("fs", "out"))

# identify spikes
rs_rates, fs_rates = [], []
for neuron in range(rs_res.shape[1]):
    signal = rs_res[int(cutoff*dts/dt):, neuron]
    signal = signal / np.max(signal)
    peaks, _ = find_peaks(signal, height=0.5, width=5)
    rs_rates.append(len(peaks)*1e3/(T-cutoff))
for neuron in range(fs_res.shape[1]):
    signal = fs_res[int(cutoff*dts/dt):, neuron]
    signal = signal / np.max(signal)
    peaks, _ = find_peaks(signal, height=0.5, width=5)
    fs_rates.append(len(peaks)*1e3/(T-cutoff))

# plot results
fig = plt.figure(figsize=(6, 6))
grid = fig.add_gridspec(nrows=4, ncols=2)

ax = fig.add_subplot(grid[0, :])
ax.plot(np.mean(rs_res, axis=1), label="RS", color="royalblue")
ax.plot(np.mean(fs_res, axis=1), label="FS", color="darkorange")
ax.set_ylabel(r'$s(t)$')
ax.set_xlabel('time (ms)')
ax.set_title('mean-field dynamics')

ax = fig.add_subplot(grid[1, :])
ax.imshow(rs_res.T, interpolation="none", aspect="auto", cmap="Greys")
ax.set_xlabel("time")
ax.set_ylabel("neuron id")
ax.set_title("RS spiking dynamics")

ax = fig.add_subplot(grid[2, :])
ax.imshow(fs_res.T, interpolation="none", aspect="auto", cmap="Greys")
ax.set_xlabel("time")
ax.set_ylabel("neuron id")
ax.set_title("FS spiking dynamics")

ax = fig.add_subplot(grid[3, 0])
ax.hist(rs_rates, density=False, rwidth=0.75)
ax.set_xlabel("spike rate")
ax.set_ylabel("count")
ax.set_title("RS spike rate histogram")

ax = fig.add_subplot(grid[3, 1])
ax.hist(fs_rates, density=False, rwidth=0.75)
ax.set_xlabel("spike rate")
ax.set_ylabel("count")
ax.set_title("FS spike rate histogram")

plt.show()

# save results
pickle.dump({'rs_results': rs_res, 'fs_results': fs_res, 'rs_rates': rs_rates, 'fs_rates': fs_rates},
            open("results/eic_snn_het_fs_bistable.p", "wb"))
