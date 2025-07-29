from rectipy import Network, random_connectivity
from pyrates import NodeTemplate, EdgeTemplate
import numpy as np
import matplotlib.pyplot as plt

# define parameters
###################

# parameters
N = 400
p = 1.0
edge_vars = {"a_ltp": 0.1, "a_ltd": 0.3, "w_max": 1.0}
Delta = 1.0
eta = 0.0
etas =  eta + Delta * np.linspace(-0.5, 0.5, num=N)
node_vars = {"tau": 1.0, "J": 5.0, "eta": etas, "tau_ltp": 10.0, "tau_ltd": 5.0, "tau_s": 0.5}
T = 200.0
dt = 1e-3
dts = 10.0*dt
I_ext = 1.0
I_start = 40.0
I_stop = 160.0

# define connectivity
W = random_connectivity(N, N, p, normalize=True)

# node and edge template initiation
node = NodeTemplate.from_yaml("config/qif_mf/qif_snn_pop")
edge = EdgeTemplate.from_yaml("config/qif_mf/stdp_edge")
for key, val in edge_vars.items():
    edge.update_var("stdp_op", key, val)

# create network
edges = {"stdp_edge/stdp_op/r_in": np.empty((N, N), dtype=np.object_),
         "stdp_edge/stdp_op/r_t": np.empty((N, N), dtype=np.object_),
         "stdp_edge/stdp_op/x_ltp": np.empty((N, N), dtype=np.object_),
         "stdp_edge/stdp_op/x_ltd": np.empty((N, N), dtype=np.object_)}
for i in range(N):
    for j in range(N):
        edges["stdp_edge/stdp_op/r_in"][i, j] = "source"
        edges["stdp_edge/stdp_op/r_t"][i, j] = f"n{i}/qif_snn_op/s"
        edges["stdp_edge/stdp_op/x_ltp"][i, j] = f"n{j}/qif_snn_op/u_ltp"
        edges["stdp_edge/stdp_op/x_ltd"][i, j] = f"n{i}/qif_snn_op/u_ltd"

# define inputs
inp = np.zeros((int(T/dt), 1))
inp[int(I_start/dt):int(I_stop/dt)] = I_ext

# run the model
###############

# initialize model
net = Network(dt, device="cpu")
net.add_diffeq_node("qif", f"config/qif_mf/qif_snn_pop", weights=W, source_var="s", target_var="s_in",
                    input_var="I_ext", output_var="s", spike_var="spike", reset_var="v", to_file=True,
                    node_vars=node_vars.copy(), op="qif_snn_op", spike_reset=-100.0, spike_threshold=100.0,
                    edge_template=edge, edge_attr=edges, clear=False)

# perform simulation
obs = net.run(inputs=inp, sampling_steps=int(dts/dt), record_output=True, verbose=True, enable_grad=False)
s = obs.to_numpy("out")
w, mapping, etas = net.get_var("qif", "w"), net.get_var("qif", "weight"), net.get_var("qif", "eta")
w, mapping, etas = w.detach().cpu().numpy(), mapping.detach().cpu().numpy(), etas.detach().cpu().numpy()
idx = np.argsort(etas)
W = np.asarray([w[mapping[i, :] > 0.0] for i in idx])
W = W[:, idx]

# plotting network connectivity
_, ax = plt.subplots(figsize=(6, 6))
step = 100
labels = np.round(etas[idx][::step], decimals=2)
im = ax.imshow(W, aspect="auto", interpolation="none", cmap="viridis")
plt.colorbar(im, ax=ax)
ax.set_xlabel("neuron")
ax.set_ylabel("neuron")
ax.set_xticks(ticks=np.arange(0, N, step=step), labels=labels)
ax.set_yticks(ticks=np.arange(0, N, step=step), labels=labels)
ax.invert_yaxis()
ax.invert_xaxis()
ax.set_title("W")
plt.tight_layout()

# plotting average firing rate dynamics
_, ax = plt.subplots(figsize=(12, 4))
s_mean = np.mean(s, axis=1)
ax.plot(s_mean, label="mean(r)")
ax.legend()
ax.set_xlabel("steps")
ax.set_ylabel("r")
plt.tight_layout()

# plotting spiking dynamics
_, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(s.T, aspect="auto", interpolation="none", cmap="Greys")
plt.colorbar(im, ax=ax)
ax.set_xlabel("steps")
ax.set_ylabel("neurons")
plt.tight_layout()

plt.show()
