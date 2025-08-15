from rectipy import Network, random_connectivity
from pyrates import NodeTemplate, EdgeTemplate
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import pickle

# define parameters
###################

# parameters
N = 200
p = 1.0
edge_vars = {"a_ltp": 0.25, "a_ltd": 0.35, "w": 0.0}
Delta = 0.2
eta = -1.5
etas_snn = eta + Delta * np.linspace(-0.5, 0.5, num=N)
snn_vars = {"tau": 1.0, "J": 5.0, "eta": etas_snn, "tau_ltp": 1.5, "tau_ltd": 1.0, "tau_s": 0.5}
T = 200.0
dt = 1e-3
dts = 1e-1
I_ext = 3.0
I_start = [50.0, 100.0, 150.0]
I_dur = 10.0

# define connectivity
W = random_connectivity(N, N, p, normalize=True)

# define inputs
inp = np.zeros((int(T/dt), 1))
for start in I_start:
    inp[int(start/dt):int((start+I_dur)/dt), 0] = I_ext

# run the snn model
###################

# node and edge template initiation
edge = EdgeTemplate.from_yaml("config/qif_mf/stdp_edge")
for key, val in edge_vars.items():
    edge.update_var("stdp_op", key, val)

# create snn network
edges_snn = {"stdp_edge/stdp_op/r_in": np.empty((N, N), dtype=np.object_),
         "stdp_edge/stdp_op/r_t": np.empty((N, N), dtype=np.object_),
         "stdp_edge/stdp_op/x_ltp": np.empty((N, N), dtype=np.object_),
         "stdp_edge/stdp_op/x_ltd": np.empty((N, N), dtype=np.object_)}
for i in range(N):
    for j in range(N):
        edges_snn["stdp_edge/stdp_op/r_in"][i, j] = "source"
        edges_snn["stdp_edge/stdp_op/r_t"][i, j] = f"n{i}/qif_snn_op/s"
        edges_snn["stdp_edge/stdp_op/x_ltp"][i, j] = f"n{j}/qif_snn_op/u_ltp"
        edges_snn["stdp_edge/stdp_op/x_ltd"][i, j] = f"n{i}/qif_snn_op/u_ltd"

# initialize model
snn_node = NodeTemplate.from_yaml("config/qif_mf/qif_snn_pop")
snn = Network(dt, device="cpu")
snn.add_diffeq_node("qif", snn_node, weights=W, source_var="s", target_var="s_in",
                    input_var="I_ext", output_var="s", spike_var="spike", reset_var="v", to_file=True,
                    node_vars=snn_vars.copy(), op="qif_snn_op", spike_reset=-100.0, spike_threshold=100.0,
                    edge_template=deepcopy(edge), edge_attr=edges_snn, clear=False)

# perform simulation
res_snn = snn.run(inputs=inp, sampling_steps=int(dts/dt), record_output=True, verbose=True, enable_grad=False)

# extract results from snn model
s = res_snn.to_numpy("out")
w, mapping, etas = snn.get_var("qif", "w"), snn.get_var("qif", "weight"), snn.get_var("qif", "eta")
w, mapping, etas = w.detach().cpu().numpy(), mapping.detach().cpu().numpy(), etas.detach().cpu().numpy()
idx = np.argsort(etas)
W = np.asarray([w[mapping[i, :] > 0.0] for i in idx])
W = W[:, idx]

# save results
##############

results = {"W": W, "s": s, "Delta": Delta}
pickle.dump(results, open(f"/home/richard-gast/Documents/results/qif_snn_Delta_{int(Delta*10.0)}.pkl", "wb"))

# plotting
##########

fig = plt.figure(figsize=(12, 4))
grid = fig.add_gridspec(nrows=1, ncols=3)

# plotting snn connectivity
ax = fig.add_subplot(grid[0, 0])
step = 100
im = ax.imshow(W, aspect="auto", interpolation="none", cmap="viridis")
plt.colorbar(im, ax=ax)
ax.set_xlabel("neuron")
ax.set_ylabel("neuron")
ax.set_xticks(ticks=np.arange(0, N, step=step))
ax.set_yticks(ticks=np.arange(0, N, step=step))
ax.invert_yaxis()
ax.invert_xaxis()
ax.set_title("Connectivity")

# plotting average firing rate dynamics
ax = fig.add_subplot(grid[0, 1:])
s_mean = np.mean(s, axis=1)
ax.plot( s_mean, label="SNN")
ax.set_xlabel("steps")
ax.set_ylabel("s")
ax.set_title("Network dynamics")

plt.tight_layout()
plt.show()
