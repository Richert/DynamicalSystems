import numpy as np
import matplotlib.pyplot as plt
from pyrates import CircuitTemplate, NodeTemplate, EdgeTemplate, clear
from numba import njit

# parameters
M = 20
edge_vars = {"a_ltp": 0.2, "a_ltd": 0.2, "w_max": 2.0}
Delta = 2.0
eta = -3.0
etas = eta + Delta * np.linspace(-0.5, 0.5, num=M)
node_vars = {"tau": 1.0, "J": 20.0 / M, "eta": etas, "tau_ltp": 2.0, "tau_ltd": 2.0, "tau_s": 0.5, "Delta": 0.5,
             "tau_a": 50.0, "kappa": 0.04}
T = 500.0
dt = 1e-3
I_ext = 3.0
I_start = 50.0
I_stop = 150.0

# node and edge template initiation
node = NodeTemplate.from_yaml("config/qif_mf/qif_stp_pop")
edge = EdgeTemplate.from_yaml("config/qif_mf/stdp_edge")
for key, val in edge_vars.items():
    edge.update_var("stdp_op", key, val)

# create network
edges = []
for i in range(M):
    for j in range(M):
        edges.append((f"p{j}/qif_stp_op/s", f"p{i}/qif_stp_op/s_in", edge,
                      {"weight": 1.0, "stdp_edge/stdp_op/r_in": f"p{j}/qif_stp_op/s", "stdp_edge/stdp_op/r_t": f"p{i}/qif_stp_op/s",
                       "stdp_edge/stdp_op/x_ltp": f"p{j}/qif_stp_op/u_ltp", "stdp_edge/stdp_op/x_ltd": f"p{i}/qif_stp_op/u_ltd",
                       }))
net = CircuitTemplate(name="qif_stp", nodes={f"p{i}": node for i in range(M)}, edges=edges)
net.update_var(node_vars={f"all/qif_stp_op/{key}": val for key, val in node_vars.items()})

# run simulation
inp = np.zeros((int(T/dt),))
inp[int(I_start/dt):int(I_stop/dt)] = I_ext
res = net.run(simulation_time=T, step_size=dt, inputs={"all/qif_stp_op/I_ext": inp},
              outputs={"r": f"all/qif_stp_op/r", "u_ltp": f"all/qif_stp_op/u_ltp", "a": f"all/qif_stp_op/a"},
              solver="heun", clear=False, sampling_step_size=0.1, decorator=njit
              )

# extract synaptic weights
mapping, weights, etas = net._ir["weight"].value, net.state["w"], net._ir["eta"].value
idx = np.argsort(etas)
W = np.asarray([weights[mapping[i, :] > 0.0] for i in idx])
W = W[:, idx]
clear(net)

# plotting
fig = plt.figure(figsize=(16, 6))
grid = fig.add_gridspec(nrows=3, ncols=3)
ax1 = fig.add_subplot(grid[0, :2])
ax1.plot(res.index, res["r"])
ax1.plot(res.index, np.mean(res["r"].values, axis=1), color="black")
ax1.set_ylabel("r (Hz)")
ax = fig.add_subplot(grid[1, :2])
ax.sharex(ax1)
ax.plot(res.index, res["u_ltp"])
ax.set_ylabel("u_ltp")
ax = fig.add_subplot(grid[2, :2])
ax.sharex(ax1)
ax.plot(res.index, res["a"])
ax.set_ylabel("a")
ax.set_xlabel("time")
ax = fig.add_subplot(grid[:, 2])
im = ax.imshow(W, aspect="auto", interpolation="none", cmap="cividis")
plt.colorbar(im, ax=ax)
step = 4
labels = np.round(etas[idx][::step], decimals=2)
ax.set_xticks(ticks=np.arange(0, M, step=step), labels=labels)
ax.set_yticks(ticks=np.arange(0, M, step=step), labels=labels)
ax.invert_yaxis()
ax.invert_xaxis()
plt.tight_layout()
plt.show()
