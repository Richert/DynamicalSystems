import numpy as np
import matplotlib.pyplot as plt
from pyrates import CircuitTemplate, NodeTemplate, EdgeTemplate, clear
from copy import deepcopy
import pickle

# parameters
M = 40
edge_vars = {"a_ltp": 0.25, "a_ltd": 0.38, "w": 0.0}
Delta = 0.1
eta = 0.0
etas = eta + Delta * np.linspace(-0.5, 0.5, num=M)
node_vars = {"tau": 1.0, "J": 5.0 / M, "eta": etas, "tau_ltp": 1.5, "tau_ltd": 1.0, "tau_s": 0.5, "Delta": 0.005}
T = 200.0
dt = 1e-3
dts = 1e-1
I_ext = 0.0
I_start = [50.0, 100.0, 150.0]
I_dur = 10.0

# node and edge template initiation
node = NodeTemplate.from_yaml("config/qif_mf/qif_stdp_pop")
edge = EdgeTemplate.from_yaml("config/qif_mf/stdp_edge")
for key, val in edge_vars.items():
    edge.update_var("stdp_op", key, val)

# create network
edges = []
for i in range(M):
    for j in range(M):
        edge_tmp = deepcopy(edge)
        edge.update_var("stdp_op", "w", np.random.uniform(0.1, 1.9))
        edges.append((f"p{j}/qif_stdp_op/s", f"p{i}/qif_stdp_op/s_in", edge,
                      {"weight": 1.0, "stdp_edge/stdp_op/r_in": f"p{j}/qif_stdp_op/s", "stdp_edge/stdp_op/r_t": f"p{i}/qif_stdp_op/s",
                       "stdp_edge/stdp_op/x_ltp": f"p{j}/qif_stdp_op/u_ltp", "stdp_edge/stdp_op/x_ltd": f"p{i}/qif_stdp_op/u_ltd",
                       }))
net = CircuitTemplate(name="qif_stdp", nodes={f"p{i}": node for i in range(M)}, edges=edges)
net.update_var(node_vars={f"all/qif_stdp_op/{key}": val for key, val in node_vars.items()})

# run simulation
inp = np.zeros((int(T/dt),))
for start in I_start:
    inp[int(start/dt):int((start+I_dur)/dt)] = I_ext
res = net.run(simulation_time=T, step_size=dt, inputs={"all/qif_stdp_op/I_ext": inp},
              outputs={"s": f"all/qif_stdp_op/s"},
              solver="heun", clear=False, sampling_step_size=dts
              )

# extract synaptic weights
mapping, weights, etas = net._ir["weight"].value, net.state["w"], net._ir["eta"].value
idx = np.argsort(etas)
W = np.asarray([weights[mapping[i, :] > 0.0] for i in idx])
W = W[:, idx]
clear(net)

# save results
##############

results = {"W": W, "res": res, "Delta": Delta}
pickle.dump(results, open(f"/home/richard-gast/Documents/results/qif_fre_Delta_{int(Delta*10.0)}.pkl", "wb"))

# plotting
##########

fig = plt.figure(figsize=(12, 4))
grid = fig.add_gridspec(nrows=1, ncols=3)

# plotting snn connectivity
ax = fig.add_subplot(grid[0, 0])
step = 8
im = ax.imshow(W, aspect="auto", interpolation="none", cmap="viridis")
plt.colorbar(im, ax=ax)
ax.set_xlabel("neuron")
ax.set_ylabel("neuron")
ax.set_xticks(ticks=np.arange(0, M, step=step))
ax.set_yticks(ticks=np.arange(0, M, step=step))
ax.invert_yaxis()
ax.invert_xaxis()
ax.set_title("Connectivity")

# plotting average firing rate dynamics
ax = fig.add_subplot(grid[0, 1:])
s_mean = np.mean(res["s"].values, axis=1)
ax.plot(res.index, s_mean, label="FRE")
ax.set_xlabel("steps")
ax.set_ylabel("s")
ax.set_title("Network dynamics")

fig.suptitle("FRE")
plt.tight_layout()
plt.show()