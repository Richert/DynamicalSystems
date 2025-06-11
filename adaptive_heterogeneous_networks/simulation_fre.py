import numpy as np
import matplotlib.pyplot as plt
from pyrates import CircuitTemplate, NodeTemplate, EdgeTemplate, clear

# parameters
M = 100
edge_vars = {"a_ltp": 0.1, "a_ltd": 0.1}
Delta = 0.5
eta = -1.5
etas = np.random.uniform(eta-Delta/2, eta+Delta/2, size=(M,))
node_vars = {"tau": 1.0, "J_e": 1.0 / M, "eta": etas, "tau_ltp": 10.0, "tau_ltd": 10.0, "Delta": Delta/M}
T = 200.0
dt = 1e-3
I_ext = 2.0
I_start = 25.0
I_stop = 175.0

# node and edge template initiation
node = NodeTemplate.from_yaml("config/qif_mf/qif_stdp_pop")
edge = EdgeTemplate.from_yaml("config/qif_mf/stdp_edge")
for key, val in edge_vars.items():
    edge.update_var("stdp_op", key, val)

# create network
edges = []
for i in range(M):
    for j in range(M):
        edges.append((f"p{j}/qif_stdp_op/r", f"p{i}/qif_stdp_op/r_e", edge,
                      {"weight": 1.0, "stdp_edge/stdp_op/r_in": "source", "stdp_edge/stdp_op/r_t": f"p{i}/qif_stdp_op/r",
                       "stdp_edge/stdp_op/x_ltp": f"p{j}/qif_stdp_op/u_ltp", "stdp_edge/stdp_op/x_ltd": f"p{i}/qif_stdp_op/u_ltp",
                       }))
net = CircuitTemplate(name="qif_stdp", nodes={f"p{i}": node for i in range(M)}, edges=edges)
net.update_var(node_vars={f"all/qif_stdp_op/{key}": val for key, val in node_vars.items()})

# run simulation
inp = np.zeros((int(T/dt),))
inp[int(I_start/dt):int(I_stop/dt)] = I_ext
res = net.run(simulation_time=T, step_size=dt, inputs={"all/qif_stdp_op/I_ext": inp},
              outputs={"r": f"all/qif_stdp_op/r", "u_ltp": f"all/qif_stdp_op/u_ltp", "u_ltd": f"all/qif_stdp_op/u_ltd"},
              solver="heun", clear=False
              )

# extract synaptic weights
mapping, weights = net._ir["weight"].value, net.state["w"]
W = np.asarray([weights[mapping[i, :] > 0.0] for i in range(M)])
clear(net)

# plotting
fig = plt.figure(figsize=(12, 6))
grid = fig.add_gridspec(nrows=3, ncols=3)
ax = fig.add_subplot(grid[0, :2])
ax.plot(res.index, res["r"])
ax.set_ylabel("r")
ax = fig.add_subplot(grid[1, :2])
ax.plot(res.index, res["u_ltp"])
ax.set_ylabel("u_ltp")
ax = fig.add_subplot(grid[2, :2])
ax.plot(res.index, res["u_ltd"])
ax.set_ylabel("u_ltd")
ax.set_xlabel("time")
ax = fig.add_subplot(grid[:, 2])
im = ax.imshow(W, aspect="auto", interpolation="none", cmap="cividis")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()
