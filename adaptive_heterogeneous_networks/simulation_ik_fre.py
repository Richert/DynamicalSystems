import numpy as np
import matplotlib.pyplot as plt
from pyrates import CircuitTemplate, NodeTemplate, EdgeTemplate, clear

# parameters
M = 20
edge_vars = {"a_ltp": 0.17, "a_ltd": 0.2}
Delta = 2.0
v_t = -40.0
thresholds = np.random.uniform(v_t-Delta/2, v_t+Delta/2, size=(M,))
node_vars = {"C": 100.0, "g": 15.0 / M, "eta": 40.0, "tau_ltp": 200.0, "tau_ltd": 100.0, "Delta": Delta/M,
             "k": 0.7, "v_r": -60.0, "v_t": thresholds, "E_r": 0.0, "b": -2.0, "tau_a": 50.0, "kappa": 80.0}
T = 2000.0
dt = 1e-3
I_ext = 60.0
I_start = 200.0
I_stop = 1800.0

# node and edge template initiation
node = NodeTemplate.from_yaml("config/qif_mf/ik_stdp_pop")
edge = EdgeTemplate.from_yaml("config/qif_mf/stdp_edge")
for key, val in edge_vars.items():
    edge.update_var("stdp_op", key, val)

# create network
edges = []
for i in range(M):
    for j in range(M):
        edges.append((f"p{j}/ik_stdp_op/r", f"p{i}/ik_stdp_op/r_in", edge,
                      {"weight": 1.0, "stdp_edge/stdp_op/r_in": "source", "stdp_edge/stdp_op/r_t": f"p{i}/ik_stdp_op/r",
                       "stdp_edge/stdp_op/x_ltp": f"p{j}/ik_stdp_op/u_ltp", "stdp_edge/stdp_op/x_ltd": f"p{i}/ik_stdp_op/u_ltp",
                       }))
net = CircuitTemplate(name="ik_stdp", nodes={f"p{i}": node for i in range(M)}, edges=edges)
net.update_var(node_vars={f"all/ik_stdp_op/{key}": val for key, val in node_vars.items()})

# run simulation
inp = np.zeros((int(T/dt),))
inp[int(I_start/dt):int(I_stop/dt)] = I_ext
res = net.run(simulation_time=T, step_size=dt, inputs={"all/ik_stdp_op/I_ext": inp},
              outputs={"r": f"all/ik_stdp_op/r", "a": f"all/ik_stdp_op/a",
                       "u_ltp": f"all/ik_stdp_op/u_ltp", "u_ltd": f"all/ik_stdp_op/u_ltd"},
              solver="heun", clear=False, sampling_step_size=10*dt
              )

# extract synaptic weights
mapping, weights, thresholds = net._ir["weight"].value, net.state["w"], net._ir["v_t"].value
idx = np.argsort(thresholds)
W = np.asarray([weights[mapping[i, :] > 0.0] for i in idx])
# clear(net)

# plotting
fig = plt.figure(figsize=(16, 6))
grid = fig.add_gridspec(nrows=4, ncols=3)
ax = fig.add_subplot(grid[0, :2])
ax.plot(res.index, res["r"]*1e3)
ax.plot(res.index, np.mean(res["r"].values, axis=1) * 1e3, color="black")
ax.set_ylabel("r (Hz)")
ax = fig.add_subplot(grid[1, :2])
ax.plot(res.index, res["u_ltp"])
ax.set_ylabel("u_ltp")
ax = fig.add_subplot(grid[2, :2])
ax.plot(res.index, res["u_ltd"])
ax.set_ylabel("u_ltd")
ax = fig.add_subplot(grid[3, :2])
ax.plot(res.index, res["a"])
ax.set_ylabel("a (mV)")
ax.set_xlabel("time")
ax = fig.add_subplot(grid[:, 2])
im = ax.imshow(W, aspect="auto", interpolation="none", cmap="cividis")
plt.colorbar(im, ax=ax)
step = 4
labels = np.round(thresholds[idx][::step], decimals=2)
ax.set_xticks(ticks=np.arange(0, M, step=step), labels=labels)
ax.set_yticks(ticks=np.arange(0, M, step=step), labels=labels)
plt.tight_layout()
plt.show()
