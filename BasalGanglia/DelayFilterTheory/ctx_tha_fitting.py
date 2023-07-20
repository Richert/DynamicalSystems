import numpy as np
import matplotlib.pyplot as plt
from rectipy import FeedbackNetwork, random_connectivity


# network parameters
n_ctx = 200
n_tha = 20
n_in = 5
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
