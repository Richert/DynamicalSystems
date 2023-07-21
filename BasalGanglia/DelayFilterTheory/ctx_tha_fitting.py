import numpy as np
import matplotlib.pyplot as plt
from rectipy import FeedbackNetwork, random_connectivity


def training_data(stim_amp: float, stim_dur: float, f1: float, f2: float, n: int, min_isi: int, max_isi: int,
                  dt: float = 1e-2) -> tuple:

    # create time variables
    T = 2.0/f1
    steps = int(T/dt)
    t = np.arange(0, steps) * dt

    # create single input-target pair
    target = np.sin(2.0*np.pi*f1*t) * np.sin(2.0*np.pi*f2*t)
    inp = np.zeros((steps, 1))
    inp[:int(stim_dur/dt), 0] = stim_amp

    # create multi-trial input-target vectors
    starts = np.random.randint(low=min_isi, high=max_isi, size=(n,))
    inputs = np.zeros((int(n*steps + np.sum(starts)), 1))
    targets = np.zeros((int(n*steps + np.sum(starts)),))
    for i in range(n):
        start = i * steps + starts[i]
        stop = (i+1) * steps + starts[i]
        inputs[start:stop] = inp
        targets[start:stop] = target

    return inputs, targets


# network parameters
n_ctx = 200
n_tha = 20
n_in = 1
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

# add network edges
net.add_edge("inp", "tha", weights=np.random.rand(n_in, n_tha))
net.add_edge("tha", "ctx", weights=k*W_ct, train="gd")
net.add_edge("ctx", "tha", weights=k*W_tc, train="gd", feedback=True)

# perform simulation of dynamics
################################

# define input
steps = 10000
inp = np.zeros((steps, n_in)) + 0.0
inp[3000:6000] += 0.2

# perform simulation
obs = net.run(inp, sampling_steps=10, enable_grad=False, record_vars=[("ctx", "v", False), ("tha", "v", False)])

# plot results
fig, axes = plt.subplots(nrows=3, figsize=(12, 9))
ax = axes[0]
ax.plot(obs.to_numpy(("ctx", "v")))
ax.set_title("ctx")
ax.set_ylabel("v")
ax = axes[1]
ax.plot(obs.to_numpy(("tha", "v")))
ax.set_title("tha")
ax.set_ylabel("v")
ax = axes[2]
ax.plot(np.mean(obs.to_numpy("out"), axis=1))
ax.set_title("ctx")
ax.set_ylabel("s")
plt.show()

# train network to generate a target function
#############################################

# get inputs and targets
inputs, targets = training_data(40.0, 10.0, 0.006, 0.012, 40, 10000, 50000, dt)

# perform weight optimization
obs = net.fit_bptt(inputs, targets)
