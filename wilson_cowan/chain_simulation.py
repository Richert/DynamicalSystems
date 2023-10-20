from pyrates import CircuitTemplate, NodeTemplate
import numpy as np
import matplotlib.pyplot as plt

# preparations
##############

# condition
cond = "chain"

# parameters
N = 10
W1 = np.eye(N)
W2 = np.zeros((N, N))
for i in range(N-1):
    W2[i+1, i] = 1.0
if cond == "ring":
    W2[0, N-1] = 1.0

k_ee = 16.0
k_ei = -15.0
k_ie = 12.0
k_ii = -3.0
k_net = 10.0
k_ee_net = 0.5*k_net
k_ie_net = 0.5*k_net
i_params = {"lam": 2.0, "phi": 3.7, "tau": 4.0, "s": 1.0}

# create circuit
pop = NodeTemplate.from_yaml("wc/E")
nodes = {f"e_{i}": pop for i in range(N)}
nodes.update({f"i_{i}": pop for i in range(N)})
e_nodes = [key for key in nodes if "e_" in key]
i_nodes = [key for key in nodes if "i_" in key]
net = CircuitTemplate("wc", nodes=nodes)
net.update_var(node_vars={f"{key}/wc_e/{var}": val for key in i_nodes for var, val in i_params.items()})

# add edges
net.add_edges_from_matrix("wc_e/u", "wc_e/u_in", weight=W1*k_ee,
                          source_nodes=e_nodes, target_nodes=e_nodes)
net.add_edges_from_matrix("wc_e/u", "wc_e/u_in", weight=W1*k_ei,
                          source_nodes=i_nodes, target_nodes=e_nodes)
net.add_edges_from_matrix("wc_e/u", "wc_e/u_in", weight=W1*k_ie,
                          source_nodes=e_nodes, target_nodes=i_nodes)
net.add_edges_from_matrix("wc_e/u", "wc_e/u_in", weight=W1*k_ii,
                          source_nodes=i_nodes, target_nodes=i_nodes)
net.add_edges_from_matrix("wc_e/u", "wc_e/u_in", weight=W2*k_ee_net,
                          source_nodes=e_nodes, target_nodes=e_nodes)
net.add_edges_from_matrix("wc_e/u", "wc_e/u_in", weight=W2*k_ie_net,
                          source_nodes=e_nodes, target_nodes=i_nodes)

# simulation
############

# input definition
t_scale = 0.3
T = 1000.0*t_scale
dt = 1e-3
dts = 1e-1
inp = np.zeros((int(T/dt), N)) + 2.0
start = 200.0*t_scale
inp[int(start/dt):, :] += 0.5

# perform simulation
res = net.run(simulation_time=T, step_size=dt, sampling_step_size=dts, solver="scipy", method="DOP853",
              atol=1e-6, rtol=1e-6, inputs={f"{key}/wc_e/s": inp[:, i] for i, key in enumerate(e_nodes)},
              outputs={"u": "all/wc_e/u"})
u = res["u"]
u_e = np.asarray([u[(n, "wc_e/u")].values for n in e_nodes])

# plotting
fig, ax = plt.subplots(figsize=(12, 5))
ax.imshow(u_e, aspect="auto", interpolation="none")
ax.set_xlabel("time")
ax.set_ylabel("pos")
plt.tight_layout()
plt.show()
