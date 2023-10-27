from rectipy import FeedbackNetwork
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# define parameters
###################

device = "cuda:0"

# network parameters
n_high = 200
n_low = 5
n_readout = 1
p = 0.2

# input parameters
n_epochs = 40
alpha = 10.0
sigma = 10
freq = 4.0
T = 1e3/freq
T_init = 500.0
dt = 5e-3
dts = 1e-1
cycle_steps = int(T/dt)
stim_onsets = np.linspace(0, T, num=n_epochs+1)[:-1]
stim_phases = 2.0*np.pi*stim_onsets/T
stim_onsets = [int(onset/dt) for onset in stim_onsets]
stim_width = int(20.0/dt)
# test_trials = list(np.arange(0, n_stims, n_tests))
# train_trials = list(np.arange(0, n_stims))
# for t in test_trials:
#     train_trials.pop(train_trials.index(t))

# define inputs and targets
# delay = 1500
# target = np.zeros((cycle_steps,))
# target[delay] = 1.0
# target = gaussian_filter1d(target, sigma=int(delay*0.1))
t = np.linspace(0, T*1e-3, cycle_steps)
f1 = 6.0
f2 = 12.0
target = np.sin(2.0*np.pi*f1*t) * np.sin(2.0*np.pi*f2*t)
idx = np.random.choice(n_low)
inputs, targets = [], []
for onset in stim_onsets:
    inp = np.zeros((onset + cycle_steps, n_low))
    targ = np.zeros((onset + cycle_steps, 1))
    inp[onset:onset + stim_width, idx] = alpha
    inputs.append(gaussian_filter1d(inp, sigma=sigma, axis=0))
    targ[onset:, 0] = target
    targets.append(targ)

# define connectivity
W_hh = np.random.randn(n_high, n_high)
W_hl = np.random.randn(n_high, n_low)
W_lh = np.random.randn(n_low, n_high)
k = 1.0
k_hh = 0.5*k
k_hl = 1.0*k
k_lh = 1.0*k

# initialize network
####################

net = FeedbackNetwork(dt=dt, device=device)

# add nodes
net.add_diffeq_node("rnn", "config/ik_snn/rate", input_var="s_in", output_var="s", source_var="s",
                    target_var="s_rec", weights=W_hh*k_hh, op="rate_op")
net.add_diffeq_node("lr", "config/ik_snn/rate", input_var="s_in", output_var="s", op="rate_op", N=n_low)
net.add_func_node("readout", n_readout, activation_function="identity")

# add edges
net.add_edge("lr", "rnn", train=None, weights=W_hl*k_hl)
# net.add_edge("rnn", "lr", train="gd", weights=W_lh*k_lh, feedback=True)
net.add_edge("rnn", "readout", train="gd")

# perform training
##################

# perform initial simulation
obs = net.run(inputs[0], sampling_steps=int(dts/dt), enable_grad=False, verbose=False,
              record_vars=[("rnn", "out", False), ("lr", "out", False)])
rnn0 = obs.to_numpy(("rnn", "out"))
lr0 = obs.to_numpy(("lr", "out"))
out0 = obs.to_numpy("out")
w0 = net.get_edge("lr", "rnn").weights.cpu().detach().numpy()

# perform fitting
loss = net.fit_bptt(inputs, targets, loss="mse", optimizer="adadelta", optimizer_kwargs={"rho": 0.9, "eps": 1e-6},
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
ax.plot(targets[0][::int(dts/dt)], color="royalblue", label="target")
ax.plot(out0, color="darkorange", label="prediction")
ax.legend()
ax.set_xlabel("time")
ax.set_ylabel("winner")
ax.set_title("Original readout channel")

# original readout dynamics
ax = fig.add_subplot(grid[4, :])
ax.plot(targets[0][::int(dts/dt)], color="royalblue", label="target")
ax.plot(out1, color="darkorange", label="prediction")
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