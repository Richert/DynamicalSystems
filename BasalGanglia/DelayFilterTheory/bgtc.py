import numpy as np
from rectipy import Network, random_connectivity, input_connections
from pyrates import CircuitTemplate, NodeTemplate, OperatorTemplate
import matplotlib.pyplot as plt

# model setup
#############

# model parameters
n_tha = 10
n_ctx = 200
n_bg = 50
tau = 10.0
k = 0.5
p = 0.2

# network connections
W_cc = input_connections(n_ctx, n_ctx, p, zero_mean=True)
W_ct = input_connections(n_ctx, n_tha, p, zero_mean=True)
W_tc = input_connections(n_tha, n_ctx, p, zero_mean=True)
W_bc = input_connections(n_bg, n_ctx, p, zero_mean=True)
W_bt = input_connections(n_bg, n_tha, p, zero_mean=True)
W_tb = random_connectivity(n_tha, n_bg, p, normalize=False)

# neuron equations
rate_op = OperatorTemplate(
    name="rate_op",
    equations="x' = -x/tau + sigmoid(k*I_syn + I_ext)",
    variables={"x": "output(0.0)", "tau": tau, "k": k, "I_ext": "input", "I_syn": "input"}
)
rate_neuron = NodeTemplate(
    name="neuron",
    operators=[rate_op]
)

# create cortical network
ctx_neurons = [f"n{i}" for i in range(n_ctx)]
ctx = CircuitTemplate(
    name="ctx",
    nodes={key: rate_neuron for key in ctx_neurons}
)
ctx.add_edges_from_matrix("rate_op/x", "rate_op/I_syn", ctx_neurons, weight=W_cc)

# create thalamus network
tha_neurons = [f"n{i}" for i in range(n_tha)]
tha = CircuitTemplate(
    name="tha",
    nodes={key: rate_neuron for key in tha_neurons}
)

# create bg network
bg_neurons = [f"n{i}" for i in range(n_bg)]
bg = CircuitTemplate(
    name="bg",
    nodes={key: rate_neuron for key in bg_neurons}
)

# combine ctx, tha and str into single network
net = CircuitTemplate(name="net", circuits={"ctx": ctx, "tha": tha, "bg": bg})
net.add_edges_from_matrix("rate_op/x", "rate_op/I_syn", source_nodes=[f"tha/{key}" for key in tha_neurons],
                          target_nodes=[f"ctx/{key}" for key in ctx_neurons],
                          weight=W_ct)
net.add_edges_from_matrix("rate_op/x", "rate_op/I_syn", source_nodes=[f"ctx/{key}" for key in ctx_neurons],
                          target_nodes=[f"tha/{key}" for key in tha_neurons],
                          weight=W_tc)
net.add_edges_from_matrix("rate_op/x", "rate_op/I_syn", source_nodes=[f"ctx/{key}" for key in ctx_neurons],
                          target_nodes=[f"bg/{key}" for key in bg_neurons],
                          weight=W_bc)
net.add_edges_from_matrix("rate_op/x", "rate_op/I_syn", source_nodes=[f"tha/{key}" for key in tha_neurons],
                          target_nodes=[f"bg/{key}" for key in bg_neurons],
                          weight=W_bt)
net.add_edges_from_matrix("rate_op/x", "rate_op/I_syn", source_nodes=[f"bg/{key}" for key in bg_neurons],
                          target_nodes=[f"tha/{key}" for key in tha_neurons],
                          weight=-W_tb)

# simulation
############

# parameters
T = 1000.0
dt = 1e-2
steps = int(T/dt)
in_neurons = [3, 6, 8]
in_times = [(10000, 30000), (40000, 60000), (70000, 90000)]
alphas = [2.0, 1.0, 4.0]

# extrinsic input
I_ext = np.zeros((n_tha, steps))
for idx, (start, stop), alpha in zip(in_neurons, in_times, alphas):
    I_ext[idx, start:stop] = alpha

# simulation
res = net.run(T, dt, inputs={f"tha/{key}/rate_op/I_ext": I_ext[idx, :] for idx, key in enumerate(tha_neurons)},
              outputs={"ctx": "ctx/all/rate_op/x", "tha": "tha/all/rate_op/x", "bg": "bg/all/rate_op/x"},
              solver="scipy", method="RK45", atol=1e-7, rtol=1e-6)

fig, axes = plt.subplots(nrows=2, figsize=(12, 7))

ax = axes[0]
ax.plot(np.mean(res["ctx"], axis=1), label="ctx")
ax.plot(np.mean(res["tha"], axis=1), label="tha")
ax.plot(np.mean(res["bg"], axis=1), label="bg")
ax.set_xlabel("time")
ax.set_ylabel("x")
ax.set_title("Mean dynamics")
ax.legend()

ax = axes[1]
ax.plot(res["tha"])
ax.set_xlabel("time")
ax.set_ylabel("x")
ax.set_title("Tha")
ax.legend()

plt.tight_layout()
plt.show()
