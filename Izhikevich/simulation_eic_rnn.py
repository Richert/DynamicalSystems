from rectipy import Network, circular_connectivity, random_connectivity
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
N = 2000
p = 0.2
v_spike = 1e3
v_reset = -1e3

# RS neuron parameters
Ce = 100.0   # unit: pF
ke = 0.7  # unit: None
ve_r = -60.0  # unit: mV
ve_t = -40.0  # unit: mV
Delta_e = 0.5  # unit: mV
de = 100.0
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
Ii = 45.0

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
inp = np.zeros((int(T/dt), 2*N))
inp[:int(0.5*cutoff/dt), N:] += 20.0
inp[int(750/dt):int(2000/dt), N:] -= 20.0

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

# connect RS and FS into a single circuit
eic = CircuitTemplate("eic", circuits={"rs": rs_net, "fs": fs_net})
eic.add_edges_from_matrix("ik_op/s", "ik_op/s_i", weight=W_ei * k_ei,
                          source_nodes=[f"fs/fs_{i}" for i in range(N)], target_nodes=[f"rs/rs_{i}" for i in range(N)])
eic.add_edges_from_matrix("ik_op/s", "ik_op/s_e", weight=W_ie * k_ie,
                          source_nodes=[f"rs/rs_{i}" for i in range(N)], target_nodes=[f"fs/fs_{i}" for i in range(N)])

# initialize rectipy model
model = Network(dt=dt, device="cuda:0")
model.add_diffeq_node("eic", node=eic, input_var="I_ext", output_var="s",
                      spike_var="spike", spike_def="v", spike_reset=v_reset,
                      spike_threshold=v_spike, verbose=True, clear=True, op="ik_op")
# model.add_edge("fs", "rs", weights=W_ei * k_ei, train=None, feedback=False)
# model.add_edge("rs", "fs", weights=W_ie * k_ie, train=None, feedback=True)

# perform simulation
obs = model.run(inputs=inp, sampling_steps=int(dts/dt), record_output=True, verbose=True)
res = obs.to_numpy("out")
rs_res = np.mean(res[:, :N], axis=1)
fs_res = np.mean(res[:, N:], axis=1)

# plot results
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot()
ax.plot(rs_res, color="royalblue", label="rs")
ax.plot(fs_res, color="darkorange", label="fs")
ax.set_ylabel(r'$s(t)$')
ax.set_xlabel('time (ms)')
ax.set_title('mean-field dynamics')
ax.legend()
plt.show()

# save results
pickle.dump({'rs_results': rs_res, 'fs_results': fs_res},
            open("results/eic_snn_het_fs_oscillatory.p", "wb"))
