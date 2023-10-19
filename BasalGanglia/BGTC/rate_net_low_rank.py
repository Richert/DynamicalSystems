from rectipy import FeedbackNetwork
import numpy as np
import matplotlib.pyplot as plt

# define parameters
###################

device = "cuda:0"

# network parameters
n_high = 500
n_low = 50
n_readout = 3
p = 0.2

# simulation parameters
t_scale = 1.0
T = 200.0 * t_scale
cutoff = 200.0 * t_scale
dt = 1e-2
dts = 1.0
input_strength = 1.0
cutoff_steps = int(cutoff/dt)
inp_steps = int(T/dt)
trial_steps = cutoff_steps + inp_steps
update_steps = int(100/dt)
n_epochs = 10

# define connectivity
W_hh = np.random.randn(n_high, n_high)
W_hl = np.random.randn(n_high, n_low)
W_lh = np.random.randn(n_low, n_high)
k = 1.0
k_hh = 0.5*k
k_hl = 1.0*k

# initialize network
####################

net = FeedbackNetwork(dt=dt, device=device)

# add nodes
net.add_diffeq_node("rnn", "config/ik_snn/rate", input_var="s_in", output_var="s", source_var="s",
                    target_var="s_rec", weights=W_hh*k_hh, op="rate_op")
net.add_diffeq_node("lr", "config/ik_snn/rate", input_var="s_in", output_var="s", op="rate_op", N=n_low)
net.add_func_node("readout", n_readout, activation_function="softmax")

# add edges
net.add_edge("lr", "rnn", train="gd", weights=W_hl*k_hl)
# net.add_edge("rnn", "lr", train=None, weights=W_lh*k_lh, feedback=True)
net.add_edge("rnn", "readout", train="gd")

# compile
net.compile()

# perform training
##################

# define input weights
input_weights = {}
n_inp = n_readout - 1
for i in range(n_inp):
    input_weights[i] = np.random.randn(n_low)
inputs, targets = [], []
for epoch in range(n_epochs):

    # define inputs and targets
    inp_seq = np.random.permutation(n_inp)
    inp = np.zeros((int(n_inp*trial_steps), n_low))
    targs = np.zeros((inp.shape[0], n_readout))
    for i, idx in enumerate(inp_seq):
        inp[i*trial_steps:i*trial_steps+inp_steps, :] = input_weights[idx] * input_strength
        targs[i*trial_steps:i*trial_steps+inp_steps, idx] = 1.0
        targs[i*trial_steps+inp_steps:i*trial_steps+inp_steps+cutoff_steps, n_inp] = 1.0

    # save inputs and targets
    inputs.append(inp)
    targets.append(targs)

# perform initial simulation
obs = net.run(inputs[0], sampling_steps=int(dts/dt), enable_grad=False, verbose=False,
              record_vars=[("rnn", "out", False), ("lr", "out", False)])
rnn0 = obs.to_numpy(("rnn", "out"))
lr0 = obs.to_numpy(("lr", "out"))
out0 = obs.to_numpy("out")
w0 = net.get_edge("lr", "rnn").weights.cpu().detach().numpy()

# perform fitting
loss = net.fit_bptt(inputs, targets, loss="mse", optimizer="adadelta", optimizer_kwargs={"rho": 0.9, "eps": 1e-5},
                    lr=0.1)

# perform final simulation
obs = net.run(inputs[0], sampling_steps=int(dts/dt), enable_grad=False, verbose=False,
              record_vars=[("rnn", "out", False), ("lr", "out", False)])
rnn1 = obs.to_numpy(("rnn", "out"))
lr1 = obs.to_numpy(("lr", "out"))
out1 = obs.to_numpy("out")
w1 = net.get_edge("lr", "rnn").weights.cpu().detach().numpy()

# plotting
##########

fig = plt.figure(figsize=(12, 9))
grid = fig.add_gridspec(nrows=6, ncols=2)

# original connectivity
ax = fig.add_subplot(grid[0, 0])
ax.imshow(w0, interpolation="none", aspect="auto")
ax.set_title("Original intrinsic weights")

# fitted connectivity
ax = fig.add_subplot(grid[0, 1])
ax.imshow(w1, interpolation="none", aspect="auto")
ax.set_title("Fitted intrinsic weights")

# original network dynamics
ax = fig.add_subplot(grid[1, :])
rnn_rates = np.mean(rnn0, axis=1)
lr_rates = np.mean(lr0, axis=1)
ax.plot(rnn_rates, color="royalblue", label="RNN")
ax.plot(lr_rates, color="darkorange", label="LR")
ax.legend()
ax.set_ylabel("s")
ax.set_title("Original network dynamics")

# fitted network dynamics
ax = fig.add_subplot(grid[2, :])
rnn_rates = np.mean(rnn1, axis=1)
lr_rates = np.mean(lr1, axis=1)
ax.plot(rnn_rates, color="royalblue", label="RNN")
ax.plot(lr_rates, color="darkorange", label="LR")
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
loss_hist = loss["epoch_loss"]
ax.plot(loss_hist)
ax.set_xlabel("epoch")
ax.set_ylabel("mse")
ax.set_title("Training loss")

plt.tight_layout()
plt.show()