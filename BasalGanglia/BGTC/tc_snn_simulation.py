from rectipy import Network, circular_connectivity, random_connectivity
from pyrates import CircuitTemplate, NodeTemplate
from scipy.stats import cauchy
import numpy as np
import matplotlib.pyplot as plt


def lorentzian(n: int, eta: float, delta: float, lb: float, ub: float):
    samples = np.zeros((n,))
    for i in range(n):
        s = cauchy.rvs(loc=eta, scale=delta)
        while s <= lb or s >= ub:
            s = cauchy.rvs(loc=eta, scale=delta)
        samples[i] = s
    return samples


# define parameters
###################

# general parameters
device = "cuda:0"
n_e = 800
n_i = 200
n_t = 100
p = 0.3
v_spike = 1e3
v_reset = -1e3
n_readout = 4

# RS neuron parameters
C_e = 100.0   # unit: pF
k_e = 0.7  # unit: None
v_r_e = -60.0  # unit: mV
v_t_e = -40.0  # unit: mV
Delta_e = 0.5  # unit: mV
d_e = 100.0  # unit: pA
a_e = 0.03  # unit: 1/ms
b_e = -2.0  # unit: nS
I_e = 70.0  # unit: pA

# LTS neuron parameters
C_i = 100.0   # unit: pF
k_i = 1.0  # unit: None
v_r_i = -56.0  # unit: mV
v_t_i = -42.0  # unit: mV
Delta_i = 1.0  # unit: mV
d_i = 20.0  # unit: pA
a_i = 0.03  # unit: 1/ms
b_i = 8.0  # unit: nS
I_i = 40.0  # unit: pA

# Tha neuron parameters
C_t = 200.0   # unit: pF
k_t = 1.6  # unit: None
v_r_t = -60.0  # unit: mV
v_t_t = -50.0  # unit: mV
Delta_t = 0.1  # unit: mV
d_t = 100.0  # unit: pA
a_t = 0.1  # unit: 1/ms
b_t = 15.0  # unit: nS
I_t = 100.0  # unit: pA

# synaptic parameters
E_ampa = 0.0
E_gaba = -65.0
tau_ampa = 10.0
tau_gaba = 20.0
k = 10.0
k_ee = 0.8*k
k_ei = 0.5*k
k_et = 1.0*k
k_ie = 0.5*k
k_ii = 0.2*k
k_it = 0.5*k
k_te = 0.5*k

# define lorentzian of etas
spike_thresholds_e = lorentzian(n_e, eta=v_t_e, delta=Delta_e, lb=v_r_e, ub=2*v_t_e - v_r_e)
spike_thresholds_i = lorentzian(n_i, eta=v_t_i, delta=Delta_i, lb=v_r_i, ub=2*v_t_i - v_r_i)
spike_thresholds_t = lorentzian(n_t, eta=v_t_t, delta=Delta_t, lb=v_r_t, ub=2*v_t_t - v_r_t)

# define connectivity
W_ee = random_connectivity(n_e, n_e, p, normalize=True)
W_ei = random_connectivity(n_e, n_i, p, normalize=True)
W_et = random_connectivity(n_e, n_t, p, normalize=True)
W_ie = random_connectivity(n_i, n_e, p, normalize=True)
W_ii = random_connectivity(n_i, n_i, p, normalize=True)
W_it = random_connectivity(n_i, n_t, p, normalize=True)
W_te = random_connectivity(n_t, n_e, p, normalize=True)

# input parameters
# t_scale = 1.0
# T = 500.0 * t_scale
# cutoff = 500.0 * t_scale
# dt = 1e-2
# dts = 1.0
# input_strength = 50.0
# cutoff_steps = int(cutoff/dt)
# inp_steps = int(T/dt)
# trial_steps = cutoff_steps + inp_steps
#
# # define inputs
# input_weights = {}
# for i in range(n_readout-1):
#     input_weights[i] = 1.0 #np.random.randn(n_t)
# inp_seq = np.random.permutation(n_readout-1)
# inp = np.zeros((int((n_readout-1)*trial_steps), n_e + n_i + n_t))
# for i, idx in enumerate(inp_seq):
#     inp[i*trial_steps:i*trial_steps+inp_steps, (n_e + n_i):] = input_weights[idx] * input_strength
T = 3000.0
dt = 1e-2
dts = 1.0
start = 1000.0
stop = 2000.0
inp = np.zeros((int(T/dt), n_e + n_i + n_t))
inp[int(start/dt):int(stop/dt), n_e + n_i:] = 50.0

# create the model
##################

# initialize EIC node
pc_vars = {"C": C_e, "k": k_e, "v_r": v_r_e, "v_t": spike_thresholds_e, "eta": I_e, "a": a_e,
           "b": b_e, "d": d_e, "tau_s": tau_ampa, "v": v_t_e, "E_e": E_ampa, "E_i": E_gaba}
in_vars = {"C": C_i, "k": k_i, "v_r": v_r_i, "v_t": spike_thresholds_i, "eta": I_i, "a": a_i,
           "b": b_i, "d": d_i, "tau_s": tau_gaba, "v": v_t_i, "E_e": E_ampa, "E_i": E_gaba}
tc_vars = {"C": C_t, "k": k_t, "v_r": v_r_t, "v_t": spike_thresholds_t, "eta": I_t, "a": a_t,
           "b": b_t, "d": d_t, "tau_s": tau_ampa, "v": v_t_t, "E_e": E_ampa, "E_i": E_gaba}
neuron_params = {"rs": pc_vars, "lts": in_vars, "tc": tc_vars}

# construct EI circuit
neurons = {"rs": n_e, "lts": n_i, "tc": n_t}
node = NodeTemplate.from_yaml(f"config/ik_snn/ik")
net_nodes = {}
for neuron in list(neurons.keys()):
    nodes = {f"{neuron}_{i}": node for i in range(neurons[neuron])}
    neurons[neuron] = list(nodes.keys())
    net_nodes.update(nodes)
net = CircuitTemplate("net", nodes=net_nodes)
for neuron in neurons:
    update_vars = {f"{neuron}_{i}/ik_op/{var}": val[i] if type(val) is np.ndarray else val
                   for i in range(len(neurons[neuron])) for var, val in neuron_params[neuron].items()}
    net.update_var(node_vars=update_vars)
net.add_edges_from_matrix(source_var="ik_op/s", target_var="ik_op/s_e", weight=W_ee*k_ee,
                          source_nodes=neurons["rs"], target_nodes=neurons["rs"])
net.add_edges_from_matrix(source_var="ik_op/s", target_var="ik_op/s_i", weight=W_ei*k_ei,
                          source_nodes=neurons["lts"], target_nodes=neurons["rs"])
net.add_edges_from_matrix(source_var="ik_op/s", target_var="ik_op/s_e", weight=W_et*k_et,
                          source_nodes=neurons["tc"], target_nodes=neurons["rs"])
net.add_edges_from_matrix(source_var="ik_op/s", target_var="ik_op/s_e", weight=W_ie*k_ie,
                          source_nodes=neurons["rs"], target_nodes=neurons["lts"])
net.add_edges_from_matrix(source_var="ik_op/s", target_var="ik_op/s_i", weight=W_ii*k_ii,
                          source_nodes=neurons["lts"], target_nodes=neurons["lts"])
net.add_edges_from_matrix(source_var="ik_op/s", target_var="ik_op/s_e", weight=W_it*k_it,
                          source_nodes=neurons["tc"], target_nodes=neurons["lts"])
net.add_edges_from_matrix(source_var="ik_op/s", target_var="ik_op/s_e", weight=W_te*k_te,
                          source_nodes=neurons["rs"], target_nodes=neurons["tc"])

# initialize rectipy model
model = Network(dt=dt, device=device)
model.add_diffeq_node("net", node=net, input_var="I_ext", output_var="s", spike_var="spike", spike_def="v",
                      spike_reset=v_reset, spike_threshold=v_spike, op="ik_op", verbose=True, clear=False)
model.add_func_node("readout", n_readout, "softmax")
W_out = np.random.randn(n_readout, n_e + n_i + n_t)
mask = np.zeros_like(W_out)
mask[:, :n_e] = 1.0
model.add_edge("net", "readout", train=None,
               weights=W_out * mask)

# perform simulation
####################

# simulation
obs = model.run(inputs=inp, sampling_steps=int(dts/dt), enable_grad=False, record_vars=[("net", "out", False)])

# extract data
out = obs.to_numpy("out")
sig = obs.to_numpy(("net", "out"))
pc_rates = np.mean(sig[:, :n_e], axis=1)
in_rates = np.mean(sig[:, n_e:(n_e+n_i)], axis=1)
tc_rates = np.mean(sig[:, (n_e+n_i):], axis=1)

# plotting
##########

# create figure
fig, axes = plt.subplots(figsize=(12, 6), nrows=3)

# plot input
ax = axes[0]
ax.imshow(inp[:, (n_e + n_i):].T, aspect="auto", interpolation="none")
ax.set_xlabel("time")
ax.set_ylabel("TC neuron")
ax.set_title("Input")

# plot network dynamics
ax = axes[1]
ax.plot(pc_rates, color="royalblue", label="PC")
ax.plot(in_rates, color="darkorange", label="IN")
ax.plot(tc_rates, color="forestgreen", label="TC")
ax.legend()
ax.set_xlabel("time")
ax.set_ylabel("s")
ax.set_title("RNN dynamics")

# plot readout
ax = axes[2]
ax.plot(out[:, 0], color="royalblue", label="C1")
ax.plot(out[:, 1], color="darkorange", label="C2")
ax.plot(out[:, 2], color="forestgreen", label="C3")
ax.plot(out[:, 3], color="black", label="C0")
ax.legend()
ax.set_xlabel("time")
ax.set_ylabel("out")
ax.set_title("Readout")

# finish
plt.tight_layout()
plt.show()
