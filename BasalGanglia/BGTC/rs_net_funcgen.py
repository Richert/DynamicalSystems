import matplotlib.pyplot as plt
import torch
from rectipy import Network, random_connectivity, FeedbackNetwork
import numpy as np
from scipy.stats import cauchy
from torch.nn import MSELoss
from torch.optim import Rprop, Adadelta
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
device = "cpu"

# network parameters
N = 1000
p = 0.2
v_spike = 1e3
v_reset = -1e3

# RS neuron parameters
C_e = 100.0   # unit: pF
k_e = 0.7  # unit: None
v_r_e = -60.0  # unit: mV
v_t_e = -40.0  # unit: mV
Delta_e = 0.59  # unit: mV
eta_e = 49.25  # unit: pA
d_e = 100.0  # unit: pA
a_e = 0.03  # unit: 1/ms
b_e = -2.0  # unit: nS
tau_s = 6.0  # unit: ms
E_e = 0.0  # unit: mV
E_i = -65.0  # unit: mV

# define lorentzian of spike thresholds
spike_thresholds_e = lorentzian(N, eta=v_t_e, delta=Delta_e, lb=v_r_e, ub=2*v_t_e - v_r_e)

# input parameters
freqs = [6.0, 12.0]
n_in = len(freqs)
n_out = 1

# define connectivity
k_ee = 15.0
k_in = 10.0
W_in = np.random.randn(N, n_in) * k_in
W_out = np.random.randn(n_out, N)
W_ee = random_connectivity(N, N, p, normalize=True) * k_ee

# simulation parameters
t_scale = 1.0
T_epoch = 500.0 * t_scale
T_init = [300.0*t_scale, 500.0*t_scale]
dt = 1e-2
dts = 1.0
n_epochs = 100

# setup model
#############

# collect parameters
pc_vars = {"C": C_e, "k": k_e, "v_r": v_r_e, "v_t": spike_thresholds_e, "eta": eta_e, "a": a_e,
           "b": b_e, "d": d_e, "tau_s": tau_s, "v": v_t_e, "E_e": E_e, "E_i": E_i}

# initialize network
net = FeedbackNetwork(dt=dt, device=device)

# add network nodes
net.add_diffeq_node("rnn", "config/ik_snn/ik", input_var="I_ext", output_var="s",
                    weights=W_ee, source_var="s", target_var="s_e", spike_var="spike", spike_def="v", op="ik_op", N=N,
                    node_vars=pc_vars)
net.add_func_node("inp", n_in, activation_function="sigmoid")
net.add_func_node("out", n_out, activation_function="identity")

# add network edges
net.add_edge("inp", "rnn", weights=W_in, train="gd")
net.add_edge("rnn", "out", weights=W_out, train="gd")
net.add_edge("rnn", "inp", feedback=True, train="gd")

# compile network
net.compile()

# perform optimization
######################

# setup loss
loss_fn = MSELoss()

# setup optimizer
# optim = Rprop(net.parameters(), lr=0.1, etas=(0.5, 1.2), step_sizes=(1e-3, 100.0))
optim = Adadelta(net.parameters(), lr=50.0, rho=0.99, eps=1e-6)

# run optimization
y0 = net.state
losses = []
for epoch in range(n_epochs):

    # get initial state
    init_steps = int(np.random.uniform(low=T_init[0], high=T_init[1])/dt)
    net.run(inputs=np.zeros((init_steps, n_in)), sampling_steps=init_steps, enable_grad=False, verbose=False)
    y1 = net.state

    # define input
    inp = np.zeros((int(T_epoch/dt), n_in))
    idx = np.random.randint(n_in)
    inp[:, idx] = 1.0

    # define target
    t = np.linspace(0, T_epoch * 1e-3, int(T_epoch/dt))
    target = np.zeros((int(T_epoch/dt), n_out))
    target[:, 0] = np.sin(freqs[idx]*np.pi*2.0*t)

    # run model
    obs = net.run(inputs=inp, enable_grad=True, sampling_steps=1, verbose=False)
    prediction = obs["out"]

    # calculate loss
    loss = loss_fn(stack(prediction), tensor(target, device=device))

    # optimization step
    optim.zero_grad()
    loss.backward()
    optim.step()
    # net.detach(requires_grad=True, detach_params=False)
    net.reset(y0)

    # save stuff for plotting
    losses.append(loss.item())
    print(f"Epoch #{epoch} finished. Current loss = {losses[-1]}")

# get fitted weights
ws = [p.detach().cpu().numpy() for p in net.parameters()]
w1 = ws[0] if ws[0].shape[0] == 1 else ws[1]

# get fitted model predictions
results = {"input": [], "target": [], "prediction": [], "rnn": []}
for idx in range(n_in):

    # define input
    inp = np.zeros((int(T_epoch/dt), n_in))
    inp[:, idx] = 1.0

    # define target
    t = np.linspace(0, T_epoch * 1e-3, int(T_epoch/dt))
    target = np.zeros((int(T_epoch/dt), n_out))
    target[:, 0] = np.sin(freqs[idx]*np.pi*2.0*t)

    # run model
    net.reset(y1)
    obs = net.run(inputs=inp, enable_grad=False, sampling_steps=1, verbose=False, record_vars=[("rnn", "out", False)])

    # store results
    results["input"].append(inp)
    results["target"].append(target)
    results["prediction"].append(obs.to_numpy("out"))
    results["rnn"].append(obs.to_numpy(("rnn", "out")))

# plotting
##########

# connectivity
fig, axes = plt.subplots(figsize=(12, 4), nrows=3)
for ax, w, title in zip(axes, [W_out, w1, w1-W_out], ["original parameters", "fitted parameters", "parameter change"]):
    ax.imshow(w, aspect="auto", interpolation="none")
    ax.set_title(title)
plt.tight_layout()

# fitted model predictions
fig, axes = plt.subplots(figsize=(12, 6), nrows=4)

ax = axes[0]
inp = np.vstack(results["input"])
ax.imshow(inp, aspect="auto", interpolation="none")
ax.set_xlabel("time")
ax.set_ylabel("channel")
ax.set_title("input")

ax = axes[1]
spikes = np.vstack(results["rnn"])
ax.imshow(spikes.T, interpolation="none", aspect="auto", cmap="Greys")
ax.set_xlabel("time")
ax.set_ylabel("neuron")
ax.set_title("RNN spikes")

ax = axes[2]
ax.plot(np.mean(spikes, axis=1))
ax.set_xlabel("time")
ax.set_ylabel("s")
ax.set_title("RNN firing rate")

ax = axes[3]
y = np.vstack(results["target"])
y_hat = np.vstack(results["prediction"])
ax.plot(y, label="target")
ax.plot(y_hat, label="prediction")
ax.set_xlabel("time")
ax.set_ylabel("out")
ax.set_title("readout")
ax.legend()

plt.tight_layout()
plt.show()
