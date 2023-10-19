import matplotlib.pyplot as plt
import torch
from torch import nn
from rectipy import FeedbackNetwork
import numpy as np

device = "cpu"
dtype = torch.float64

# input parameters
t_scale = 1.0
T = 100.0 * t_scale
cutoff = 100.0 * t_scale
dt = 5e-3
dts = 1.0
input_strength = 10.0
cutoff_steps = int(cutoff/dt)
inp_steps = int(T/dt)
trial_steps = cutoff_steps + inp_steps
update_steps = int(100/dt)
n_epochs = 10

# model parameters
node = "neuron_model_templates.spiking_neurons.lif.lif"
N1 = 100
N2 = 10
k = 2.0
tau1 = np.random.uniform(10.0, 20.0, size=(N1,))
tau2 = np.random.uniform(10.0, 20.0, size=(N2,))
tau_s = 5.0
eta = 10.0
v_thr = 10.0
v_reset = -10.0
node_vars1 = {"eta": eta, "tau": tau1, "tau_s": tau_s, "k": k}
node_vars2 = {"eta": eta, "tau": tau2, "tau_s": tau_s, "k": k}

# define connectivity
n_out = 3
n_in = n_out - 1
W_in = np.random.randn(N2, n_in)
J = np.random.randn(N1, N1)
W_12 = np.random.randn(N1, N2)
W_21 = np.random.randn(N2, N1)
W_out = np.random.randn(n_out, N1)

# initialize target network
net = FeedbackNetwork(dt, device=device)
net.add_diffeq_node("lif2", node=node, spike_def="v", spike_var="spike", input_var="s_in", output_var="s",
                    clear=True, float_precision="float64", op="lif_op", N=N2, node_vars=node_vars2,
                    spike_threshold=v_thr, spike_reset=v_reset)
net.add_diffeq_node("lif1", node=node, weights=J, source_var="s", spike_def="v", spike_var="spike", target_var="s_in",
                    input_var="I_ext", output_var="s", clear=True, float_precision="float64", op="lif_op",
                    node_vars=node_vars1, spike_threshold=v_thr, spike_reset=v_reset)
net.add_func_node(label="inp", n=n_in, activation_function="identity")
net.add_func_node(label="out", n=n_out, activation_function="identity")
net.add_edge("inp", "lif2", weights=W_in)
net.add_edge("lif2", "lif1", weights=W_12, train="gd")
net.add_edge("lif1", "out", weights=W_out, train="gd")
net.add_edge("lif1", "lif2", weights=W_21, train=None, feedback=True)
net.compile()

# define input weights
input_weights = {}
inputs, targets = [], []
for epoch in range(n_epochs):

    # define inputs and targets
    inp_seq = np.random.permutation(n_in)
    inp = np.zeros((int(n_in*trial_steps), n_in))
    targs = np.zeros((inp.shape[0], n_out))
    for i, idx in enumerate(inp_seq):
        inp[i*trial_steps:i*trial_steps+inp_steps, :] = input_strength
        targs[i*trial_steps:i*trial_steps+inp_steps, idx] = 1.0
        targs[i*trial_steps+inp_steps:i*trial_steps+inp_steps+cutoff_steps, n_in] = 1.0

    # save inputs and targets
    inputs.append(inp)
    targets.append(targs)

# perform initial simulation
obs = net.run(inputs[0], sampling_steps=int(dts/dt), enable_grad=False, verbose=False,
              record_vars=[("lif1", "out", False), ("lif2", "out", False)])
s1_0 = obs.to_numpy(("lif1", "out"))
s2_0 = obs.to_numpy(("lif2", "out"))
out0 = obs.to_numpy("out")
w1_0 = net.get_edge("lif2", "lif1").weights.cpu().detach().numpy()
w2_0 = net.get_edge("lif1", "out").weights.cpu().detach().numpy()

# perform fitting
loss = net.fit_bptt(inputs, targets, loss="ce", optimizer="adadelta", optimizer_kwargs={"rho": 0.9, "eps": 1e-5},
                    lr=0.1)
loss_hist = loss["epoch_loss"]

# perform final simulation
obs = net.run(inputs[0], sampling_steps=int(dts/dt), enable_grad=False, verbose=False,
              record_vars=[("lif1", "out", False), ("lif2", "out", False)])
s1_1 = obs.to_numpy(("lif1", "out"))
s2_1 = obs.to_numpy(("lif2", "out"))
out1 = obs.to_numpy("out")
w1_1 = net.get_edge("lif2", "lif1").weights.cpu().detach().numpy()
w2_1 = net.get_edge("lif1", "out").weights.cpu().detach().numpy()

# plotting
##########

fig = plt.figure(figsize=(12, 9))
grid = fig.add_gridspec(nrows=6, ncols=2)

# original connectivity
ax = fig.add_subplot(grid[0, 0])
diff = w1_0 - w1_1
ax.imshow(diff, interpolation="none", aspect="auto")
ax.set_title(f"Changes to W_12 (min: {np.min(diff)}, max: {np.max(diff)})")

# fitted connectivity
ax = fig.add_subplot(grid[0, 1])
diff = w2_0 - w2_1
ax.imshow(diff, interpolation="none", aspect="auto")
ax.set_title(f"Changes to W_12 (min: {np.min(diff)}, max: {np.max(diff)})")

# original network dynamics
ax = fig.add_subplot(grid[1, :])
l1_rates = np.mean(s1_0, axis=1)
l2_rates = np.mean(s2_0, axis=1)
ax.plot(l1_rates, color="royalblue", label="L1")
ax.plot(l2_rates, color="darkorange", label="L2")
ax.legend()
ax.set_ylabel("s")
ax.set_title("Original network dynamics")

# fitted network dynamics
ax = fig.add_subplot(grid[2, :])
l1_rates = np.mean(s1_1, axis=1)
l2_rates = np.mean(s2_1, axis=1)
ax.plot(l1_rates, color="royalblue", label="L1")
ax.plot(l2_rates, color="darkorange", label="L2")
ax.legend()
ax.set_ylabel("s")
ax.set_title("Fitted network dynamics")

# original readout dynamics
ax = fig.add_subplot(grid[3, :])
ax.plot(np.argmax(targets[0], axis=1)[::int(dts/dt)], color="royalblue", label="target")
ax.plot(np.argmax(out0, axis=1), color="darkorange", label="prediction")
ax.legend()
ax.set_xlabel("time")
ax.set_ylabel("winner")
ax.set_title("Original readout channel")

# original readout dynamics
ax = fig.add_subplot(grid[4, :])
ax.plot(np.argmax(targets[0], axis=1)[::int(dts/dt)], color="royalblue", label="target")
ax.plot(np.argmax(out1, axis=1), color="darkorange", label="prediction")
ax.legend()
ax.set_xlabel("time")
ax.set_ylabel("winner")
ax.set_title("Fitted readout channel")

# epoch loss
ax = fig.add_subplot(grid[5, :])
ax.plot(loss_hist)
ax.set_xlabel("epoch")
ax.set_ylabel("mse")
ax.set_title("Training loss")

plt.tight_layout()
plt.show()
