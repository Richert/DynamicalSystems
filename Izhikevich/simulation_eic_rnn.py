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
N = 1000
p = 0.2
v_spike = 1e3
v_reset = -1e3

# RS neuron parameters
Ce = 100.0   # unit: pF
ke = 0.7  # unit: None
ve_r = -60.0  # unit: mV
ve_t = -40.0  # unit: mV
Delta_e = 1.0  # unit: mV
de = 20.0
ae = 0.03
be = -2.0

# IB neuron parameters
Ci = 20.0   # unit: pF
ki = 1.0  # unit: None
vi_r = -55.0  # unit: mV
vi_t = -40.0  # unit: mV
Delta_i = 0.3  # unit: mV
di = 0.0
ai = 0.2
bi = 0.025

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
spike_thresholds_e = lorentzian(N, eta=ve_t, delta=Delta_e, lb=ve_r, ub=ve_r-ve_t)
spike_thresholds_i = lorentzian(N, eta=vi_t, delta=Delta_i, lb=vi_r, ub=vi_r-vi_t)

# define connectivity
indices = np.arange(0, N, dtype=np.int32)
pdfs = np.asarray([dist(idx, method="inverse", zero_val=0.0, inverse_pow=1.5) for idx in indices])
pdfs /= np.sum(pdfs)
W_ee = circular_connectivity(N, p, spatial_distribution=rv_discrete(values=(indices, pdfs)))
W_ie = circular_connectivity(N, p, spatial_distribution=rv_discrete(values=(indices, pdfs)))
W_ei = circular_connectivity(N, p, spatial_distribution=rv_discrete(values=(indices, pdfs)))
W_ii = circular_connectivity(N, p, spatial_distribution=rv_discrete(values=(indices, pdfs)))

# define inputs
T = 2500.0
cutoff = 500.0
dt = 1e-3
dts = 1e-1
inp = np.zeros((int(T/dt), 1)) + 25.0
inp[int(1000/dt):int(2000/dt), 0] += 25.0

# run the model
###############

# initialize nodes
e_vars = {"C": Ce, "k": ke, "v_r": ve_r, "v_theta": spike_thresholds_e, "eta": 0.0, "tau_u": 1/ae, "b": be, "kappa": de,
          "g_e": g_ampa, "E_e": E_ampa, "g_i": g_gaba, "E_i": E_gaba, "tau_s": tau_ampa, "v": ve_t}
i_vars = {"C": Ci, "k": ki, "v_r": vi_r, "v_theta": spike_thresholds_i, "eta": 0.0, "tau_u": 1/ai, "b": bi, "kappa": di,
          "g_e": g_ampa, "E_e": E_ampa, "g_i": g_gaba, "E_i": E_gaba, "tau_s": tau_gaba, "v": vi_t}
rs = NodeTemplate.from_yaml("config/ik_snn/ik")
fs = NodeTemplate.from_yaml("config/ik_snn/ik")

# construct rs and fs circuits
rs_neurons = {f'rs_{i}': rs for i in range(N)}
rs_net = CircuitTemplate("rs", nodes=rs_neurons)
rs_net.add_edges_from_matrix(source_var="ik_op/s", target_var="ik_op/s_e", weight=W_ee,
                             source_nodes=list(rs_neurons.keys()))
rs_net.update_var(node_vars={f"all/ik_op/{key}": val for key, val in e_vars.items()})
fs_neurons = {f'fs_{i}': fs for i in range(N)}
fs_net = CircuitTemplate("fs", nodes=fs_neurons)
fs_net.add_edges_from_matrix(source_var="ik_op/s", target_var="ik_op/s_i", weight=W_ii,
                             source_nodes=list(fs_neurons.keys()))
fs_net.update_var(node_vars={f"all/ik_op/{key}": val for key, val in i_vars.items()})

# combine the rs and fs populations into a single circuit
net = CircuitTemplate("eic", circuits={"rs": rs_net, "fs": fs_net})
net.add_edges_from_matrix(source_var="ik_op/s", target_var="ik_op/s_e", weight=W_ie,
                          source_nodes=[f"rs/rs_{i}" for i in range(N)], target_nodes=[f"fs/fs_{i}" for i in range(N)])
net.add_edges_from_matrix(source_var="ik_op/s", target_var="ik_op/s_i", weight=W_ei,
                          source_nodes=[f"fs/fs_{i}" for i in range(N)], target_nodes=[f"rs/rs_{i}" for i in range(N)])

# initialize rectipy model
model = Network(dt=dt, device="cpu")
model.add_diffeq_node("eic", node=net, input_var="I_ext", output_var="s", spike_var="spike", spike_def="v",
                      op="ik_op", spike_reset=v_reset, spike_threshold=v_spike, verbose=False, clear=True)

# perform simulation
obs = model.run(inputs=inp, sampling_steps=int(dts/dt), record_output=True, verbose=False)
res = obs.to_numpy("out")

# plot results
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(np.mean(res, axis=1))
ax.set_ylabel(r'$s(t)$')
ax.set_xlabel('time')
plt.tight_layout()
plt.show()

# save results
pickle.dump({'results': res}, open("results/rs_snn_hom_low_sfa.p", "wb"))
