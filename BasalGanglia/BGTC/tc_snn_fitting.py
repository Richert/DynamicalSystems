from rectipy import Network, circular_connectivity, random_connectivity
from pyrates import CircuitTemplate, NodeTemplate
from scipy.stats import cauchy
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import MSELoss
from torch.optim import Adadelta
from torch import stack, tensor


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
epsilon = 0.01

# input parameters
t_scale = 1.0
T = 200.0 * t_scale
cutoff = 200.0 * t_scale
dt = 1e-2
dts = 1.0
input_strength = 50.0
cutoff_steps = int(cutoff/dt)
inp_steps = int(T/dt)
trial_steps = cutoff_steps + inp_steps
update_steps = int(100/dt)
n_epochs = 10

# network parameters
n_e = 200
n_i = 40
n_t = 20
p = 0.2
v_spike = 1e3
v_reset = -1e3
n_readout = 3

# RS neuron parameters
C_e = 100.0   # unit: pF
k_e = 3.0  # unit: None
v_r_e = -60.0  # unit: mV
v_t_e = -50.0  # unit: mV
Delta_e = 0.5  # unit: mV
d_e = 400.0  # unit: pA
a_e = 0.01  # unit: 1/ms
b_e = 5.0  # unit: nS
I_e = 50.0  # unit: pA

# LTS neuron parameters
C_i = 100.0   # unit: pF
k_i = 1.0  # unit: None
v_r_i = -56.0  # unit: mV
v_t_i = -42.0  # unit: mV
Delta_i = 1.0  # unit: mV
d_i = 20.0  # unit: pA
a_i = 0.03  # unit: 1/ms
b_i = 8.0  # unit: nS
I_i = 60.0  # unit: pA

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
g_ampa = 1.0
g_gaba = 1.0
E_ampa = 0.0
E_gaba = -65.0
tau_ampa = 6.0
tau_gaba = 8.0
k_ee = 20.0
k_ei = 15.0
k_et = 10.0
k_ie = 10.0
k_ii = 10.0
k_it = 5.0
k_te = 20.0

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

# create the model
##################

# initialize EIC node
pc_vars = {"C": C_e, "k": k_e, "v_r": v_r_e, "v_theta": spike_thresholds_e, "eta": I_e, "tau_u": 1/a_e,
           "b": b_e, "kappa": d_e, "tau_s": tau_gaba, "v": v_t_e}
in_vars = {"C": C_i, "k": k_i, "v_r": v_r_i, "v_theta": spike_thresholds_i, "eta": I_i, "tau_u": 1/a_i,
           "b": b_i, "kappa": d_i, "tau_s": tau_gaba, "v": v_t_i}
tc_vars = {"C": C_t, "k": k_t, "v_r": v_r_t, "v_theta": spike_thresholds_t, "eta": I_t, "tau_u": 1/a_t,
           "b": b_t, "kappa": d_t, "tau_s": tau_gaba, "v": v_t_t}
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
                      spike_reset=v_reset, spike_threshold=v_spike, op="ik_op", verbose=True, clear=False,
                      train_params=["weights"])

# perform parameter optimization
################################

# add a readout layer
model.add_func_node("readout", n_readout, "softmax")
model.add_edge("net", "readout", train="gd")
model.compile()

# define input weights
input_weights = {}
for i in range(n_readout):
    input_weights[i] = np.random.rand(n_t)
inputs, targets = [], []
for epoch in range(n_epochs):

    # define inputs and targets
    inp_seq = np.random.permutation(n_readout)
    inp = np.zeros((int(n_readout*trial_steps), n_e + n_i + n_t))
    targs = np.zeros((inp.shape[0], n_readout))
    for i, idx in enumerate(inp_seq):
        inp[i*trial_steps:i*trial_steps+inp_steps, (n_e + n_i):] = input_weights[idx] * input_strength
        targs[i*trial_steps:i*trial_steps+inp_steps, idx] = 1.0

    # save inputs and targets
    inputs.append(inp)
    targets.append(targs)

# perform initial simulation
obs = model.run(inputs[0], sampling_steps=int(dts/dt), enable_grad=False, record_vars=[("net", "out", False)],
                record_output=False)
res0 = obs.to_numpy(("net", "out"))
w0 = model.get_var("net", "weights").cpu().detach().numpy()

# perform fitting
loss_fn = MSELoss()
optim = Adadelta(model.parameters(), lr=0.5, rho=0.9, eps=1e-6)
loss_hist = []
for epoch in range(n_epochs):

    # perform forward pass
    obs = model.run(inputs[epoch], sampling_steps=1, enable_grad=True, verbose=False)
    predictions = obs["out"]

    # calculate loss
    loss = loss_fn(stack(predictions), tensor(targets[epoch], device=device))

    # perform weight update
    optim.zero_grad()
    loss.backward()
    optim.step()
    model.detach(detach_params=False, requires_grad=True)
    model.reset(model.state)

    # save stuff for plotting
    loss_hist.append(loss.item())
    print(f"Epoch #{epoch} finished. Current loss = {loss_hist[-1]}")

    # convergence criterion
    if loss_hist[-1] < epsilon:
        break

# perform final simulation
obs = model.run(inputs[0], sampling_steps=int(dts/dt), enable_grad=False, record_vars=[("net", "out", False)],
                record_output=False)
res1 = obs.to_numpy(("net", "out"))
w1 = model.get_var("net", "weights").cpu().detach().numpy()

# plotting
##########

fig = plt.figure(figsize=(12, 8))
grid = fig.add_gridspec(nrows=4, ncols=2)

ax = fig.add_subplot(grid[0, 0])
ax.imshow(w0, interpolation="none", aspect="auto")
ax.set_title("Original intrinsic weights")

ax = fig.add_subplot(grid[0, 1])
ax.imshow(w1, interpolation="none", aspect="auto")
ax.set_title("Fitted intrinsic weights")

ax = fig.add_subplot(grid[1, :])
sig = res0
pc_rates = np.mean(sig[:, :n_e], axis=1)
in_rates = np.mean(sig[:, n_e:(n_e+n_i)], axis=1)
tc_rates = np.mean(sig[:, (n_e+n_i):], axis=1)
ax.plot(pc_rates, color="royalblue", label="PC")
ax.plot(in_rates, color="darkorange", label="IN")
ax.plot(tc_rates, color="forestgreen", label="TC")
ax.legend()
ax.set_xlabel("time")
ax.set_ylabel("s")
ax.set_title("Original network dynamics")

ax = fig.add_subplot(grid[2, :])
sig = res1
pc_rates = np.mean(sig[:, :n_e], axis=1)
in_rates = np.mean(sig[:, n_e:(n_e+n_i)], axis=1)
tc_rates = np.mean(sig[:, (n_e+n_i):], axis=1)
ax.plot(pc_rates, color="royalblue", label="PC")
ax.plot(in_rates, color="darkorange", label="IN")
ax.plot(tc_rates, color="forestgreen", label="TC")
ax.legend()
ax.set_xlabel("time")
ax.set_ylabel("s")
ax.set_title("Fitted network dynamics")

ax = fig.add_subplot(grid[3, :])
ax.plot(loss_hist)
ax.set_xlabel("epoch")
ax.set_ylabel("mse")
ax.set_title("Training loss")

plt.tight_layout()
plt.show()
