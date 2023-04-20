from rectipy import Network, circular_connectivity
from pyrates import CircuitTemplate, NodeTemplate
import sys
sys.path.append("~/PycharmProjects/DynamicalSystems/reservoir_computing")
import numpy as np
from reservoir_computing.utility_funcs import lorentzian, dist
import matplotlib.pyplot as plt
import pickle
from scipy.stats import rv_discrete


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
Delta_e = 1.0  # unit: mV
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
Ii = 75.0

# synaptic parameters
g_ampa = 1.0
g_gaba = 1.0
E_ampa = 0.0
E_gaba = -65.0
tau_ampa = 6.0
tau_gaba = 8.0
k_ee = 15.0
k_ei = 10.0
k_ie = 5.0
k_ii = 10.0

# define lorentzian of etas
spike_thresholds_e = lorentzian(N, eta=ve_t, delta=Delta_e, lb=ve_r, ub=2*ve_t - ve_r)
spike_thresholds_i = lorentzian(N, eta=vi_t, delta=Delta_i, lb=vi_r, ub=2*vi_t - vi_r)

# define connectivity
indices = np.arange(0, N, dtype=np.int32)
e_pdfs = np.asarray([dist(idx, method="inverse", zero_val=0.0, inverse_pow=1.5) for idx in indices])
e_pdfs /= np.sum(e_pdfs)
i_pdfs = np.asarray([dist(idx, method="inverse", zero_val=0.0, inverse_pow=1.5) for idx in indices])
i_pdfs /= np.sum(i_pdfs)
W_ee = circular_connectivity(N, p, spatial_distribution=rv_discrete(values=(indices, e_pdfs)), homogeneous_weights=False)
W_ie = circular_connectivity(N, p, spatial_distribution=rv_discrete(values=(indices, e_pdfs)), homogeneous_weights=False)
W_ei = circular_connectivity(N, p, spatial_distribution=rv_discrete(values=(indices, i_pdfs)), homogeneous_weights=False)
W_ii = circular_connectivity(N, p, spatial_distribution=rv_discrete(values=(indices, i_pdfs)), homogeneous_weights=False)

# define inputs
T = 2500.0
cutoff = 500.0
dt = 1e-2
dts = 1e-1
p_in = 0.25
n_inputs = int(p_in*N)
center = int(N*0.5)
inp_indices = N + np.arange(center-int(0.5*n_inputs), center+int(0.5*n_inputs))
inp = np.zeros((int(T/dt), 2*N))
inp[:int(0.5*cutoff/dt), N:] += 30.0
inp[int(1000/dt):int(2000/dt), inp_indices] -= 40.0

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
net = CircuitTemplate("eic", circuits={"rs": rs_net, "fs": fs_net})
net.add_edges_from_matrix(source_var="ik_op/s", target_var="ik_op/s_e", weight=W_ie * k_ie,
                          source_nodes=[f"rs/rs_{i}" for i in range(N)], target_nodes=[f"fs/fs_{i}" for i in range(N)])
net.add_edges_from_matrix(source_var="ik_op/s", target_var="ik_op/s_i", weight=W_ei * k_ei,
                          source_nodes=[f"fs/fs_{i}" for i in range(N)], target_nodes=[f"rs/rs_{i}" for i in range(N)])

# initialize rectipy model
model = Network(dt=dt, device="cuda:0")
model.add_diffeq_node("eic", node=net, input_var="ik_op/I_ext", output_var="rs/all/ik_op/s",
                      spike_var="ik_op/spike", spike_def="all/all/ik_op/v", spike_reset=v_reset,
                      spike_threshold=v_spike, verbose=False, clear=False)

# perform simulation
obs = model.run(inputs=inp, sampling_steps=int(dts/dt), record_output=True, verbose=False)
res = obs.to_numpy("out")

# plot results
fig, axes = plt.subplots(nrows=2, figsize=(12, 6))
ax = axes[0]
ax.plot(np.mean(res, axis=1))
ax.set_ylabel(r'$s(t)$')
ax.set_xlabel('time')
ax = axes[1]
ax.imshow(res.T, interpolation="none", aspect="auto", cmap="Greys")
ax.set_xlabel("time")
ax.set_ylabel("neuron id")
plt.tight_layout()
plt.show()

# save results
pickle.dump({'results': res}, open("results/eic_snn_het_fs_hom_rs.p", "wb"))
