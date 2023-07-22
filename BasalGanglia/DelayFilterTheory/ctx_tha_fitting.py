import numpy as np
import matplotlib.pyplot as plt
import torch
from rectipy import FeedbackNetwork, random_connectivity


def training_data(stim_amp: float, stim_dur: float, f1: float, f2: float, n: int, min_isi: int, max_isi: int,
                  dt: float = 1e-2) -> tuple:

    # create time variables
    T = 2.0/f1
    steps = int(T/dt)
    t = np.arange(0, steps) * dt

    # create single input-target pair
    target = np.sin(2.0*np.pi*f1*t) * np.sin(2.0*np.pi*f2*t)
    inp = np.zeros((steps,))
    inp[:int(stim_dur/dt)] = stim_amp

    # create multi-trial input-target vectors
    inputs = []
    targets = []
    for start in np.random.randint(low=min_isi, high=max_isi, size=(n,)):
        inp_tmp = np.zeros((steps + start, 1))
        inp_tmp[start:, 0] = inp
        target_tmp = np.zeros_like(inp_tmp)
        target_tmp[start:, 0] = target
        inputs.append(inp_tmp)
        targets.append(target_tmp)

    return inputs, targets


# network parameters
n_ctx = 200
n_tha = 20
n_in = 1
n_out = 1
p = 0.2
k = 2.0
dt = 1e-2
node = "config/ik_snn/rs"

# build network
###############

# initialize network
net = FeedbackNetwork(dt, device="cuda:0")

# network connectivity
W_cc = random_connectivity(n_ctx, n_ctx, p, normalize=True)
W_tt = np.zeros((n_tha, n_tha))
W_ct = random_connectivity(n_ctx, n_tha, p, normalize=True)
W_tc = random_connectivity(n_tha, n_ctx, p, normalize=True)

# add network nodes
net.add_diffeq_node("ctx", node, input_var="s_ext", output_var="s", weights=W_cc, source_var="s",
                    target_var="s_in", op="rs_op", spike_var="spike", spike_def="v")
net.add_diffeq_node("tha", node, input_var="s_ext", output_var="s", weights=W_tt, source_var="s",
                    target_var="s_in", op="rs_op", spike_var="spike", spike_def="v")
net.add_func_node("inp", n_in, activation_function="identity")
net.add_func_node("out", n_out, activation_function="identity")

# add network edges
net.add_edge("inp", "tha", weights=np.random.rand(n_in, n_tha))
net.add_edge("tha", "ctx", weights=k*W_ct, train="gd")
net.add_edge("ctx", "tha", weights=k*W_tc, train="gd", feedback=True)
net.add_edge("ctx", "out", weights=np.random.rand(n_ctx, n_out))

# perform simulation of dynamics
################################

# # define input
# steps = 10000
# inp = np.zeros((steps, n_in)) + 0.0
# inp[3000:6000] += 0.2
#
# # perform simulation
# obs = net.run(inp, sampling_steps=10, enable_grad=False, record_vars=[("ctx", "v", False), ("tha", "v", False)])
#
# # plot results
# fig, axes = plt.subplots(nrows=3, figsize=(12, 9))
# ax = axes[0]
# ax.plot(obs.to_numpy(("ctx", "v")))
# ax.set_title("ctx")
# ax.set_ylabel("v")
# ax = axes[1]
# ax.plot(obs.to_numpy(("tha", "v")))
# ax.set_title("tha")
# ax.set_ylabel("v")
# ax = axes[2]
# ax.plot(obs.to_numpy("out"))
# ax.set_title("readout")
# ax.set_ylabel("")
# plt.show()

# train network to generate a target function
#############################################

# get inputs and targets
inputs, targets = training_data(40.0, 10.0, 0.006, 0.012, 40, 10000, 50000, dt)

# choose sampling rate
sr = 10
targets = [t[::sr, :] for t in targets]

# perform weight optimization
obs = net.fit_bptt(inputs, targets, optimizer="rprop", optimizer_kwargs={"etas": (0.5, 1.2)},
                   sampling_steps=sr, update_steps=100000)

# get test data
in_test, tar_test = training_data(40.0, 10.0, 0.006, 0.012, 3, 10000, 50000, dt)
in_test = torch.vstack(in_test)
tar_test = torch.vstack(tar_test).numpy()
obs_test = net.run(in_test, sampling_steps=sr, enable_grad=False)

# plot results
fig, axes = plt.subplots(nrows=3, figsize=(12, 6))
ax = axes[0]
ax.plot(obs.to_numpy("epochs"), obs.to_numpy("epoch_loss"))
ax.set_title("MSE")
ax = axes[1]
ax.plot(obs_test.to_numpy("out"), label="prediction")
ax.plot(tar_test[::sr, 0], label="target")
ax.set_title("Fit")
ax.legend()
plt.show()
