import numpy as np
import matplotlib.pyplot as plt
from pyrates import NodeTemplate, CircuitTemplate
from rectipy import Network
from networkx import random_partition_graph, to_pandas_adjacency
from utility_funcs import lorentzian, get_module_coupling, fit_lorentzian


# model parameters
n_comms = 3
n_neurons = 500
N = int(n_comms*n_neurons)
p_in = 0.06
p_out = 0.01
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
Delta = 1.0
eta = 55.0
a = 0.03
b = -2.0
kappa = 100.0
g = 15.0
E_r = 0.0
tau_s = 6.0
v_spike = 1000.0
v_reset = -1000.0

# simulation parameters
cutoff = 1000.0
T = 4000.0 + cutoff
dt = 1e-2
sr = 100
steps = int(T/dt)
cutoff_steps = int(cutoff/(dt*sr))
I_ext = np.zeros((steps, 1))

# meta parameters
device = "cuda:0"

# simulate SNN dynamics
#######################

# create SNN connectivity
G = random_partition_graph([n_neurons for _ in range(n_comms)], p_in, p_out, directed=False)
W = to_pandas_adjacency(G).values
W /= np.sum(W, axis=1)
plt.imshow(W)
plt.show()

# draw spike thresholds from distribution
thetas = lorentzian(N, v_t, Delta, v_r, 2 * v_t - v_r)

# initiallize network
snn_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": thetas, "eta": eta, "tau_u": 1/a, "b": b, "kappa": kappa,
            "g": g, "E_r": E_r, "tau_s": tau_s, "v": v_t}
snn = Network.from_yaml("config/ik/rs", weights=W, source_var="s", target_var="s_in", input_var="s_ext", output_var="s",
                        spike_var="spike", spike_def="v", to_file=False, node_vars=snn_vars.copy(), op="rs_op",
                        spike_reset=v_reset, spike_threshold=v_spike, dt=dt, verbose=False, clear=True, device=device)

# simulate SNN dynamics
obs = snn.run(inputs=I_ext, sampling_steps=sr, record_output=True, verbose=False)
snn_res = obs["out"].iloc[cutoff_steps:, :]

# simulate MF dynamics
######################

# define communities
modules = {}
counter = 0
for idx in range(n_comms):
    modules[str(counter)] = [i+counter for i in range(n_neurons)]
    counter += n_neurons

# get mean-field connectivity
W_mf = get_module_coupling(W, modules)

# get mean-field spike threshold distribution parameters
deltas, thresholds = fit_lorentzian(thetas, modules)

# initialize mean-field network
node_vars = {"C": C, "k": k, "v_r": v_r, "v_t": thresholds, "Delta": deltas, "tau_u": 1/a, "b": b, "E_r": E_r,
             "eta": eta, "kappa": kappa, "g": g, "tau_s": tau_s, "v": v_t}
rs = NodeTemplate.from_yaml("config/ik/rs_mf")
nodes = [str(key) for key in modules]
mf = CircuitTemplate("rs", nodes={key: rs for key in nodes})
mf.add_edges_from_matrix("rs_mf_op/s", "rs_mf_op/s_in", nodes=nodes, weight=W_mf)
mf.update_var(node_vars={f"all/rs_mf_op/{key}": val for key, val in node_vars.items()})

# simulate mean-field network dynamics
res = mf.run(simulation_time=T, step_size=dt, sampling_step_size=dt*sr, outputs={"s": "all/rs_mf_op/s"},
             solver="scipy", method="DOP853", cutoff=cutoff)

# calculate module covariance patterns
cov_mf = np.cov(res["s"].values.T)
cov_snn = np.cov(np.asarray([np.mean(snn_res.iloc[:, idx]) for idx in modules.values()]))

# plotting
fig, axes = plt.subplots(nrows=len(modules), figsize=(10, 2*len(modules)))
for i, mod in enumerate(modules):
    ax = axes[i]
    ax.plot(res.index, np.mean(snn_res.iloc[:, modules[mod]], axis=1), label="SNN")
    ax.plot(res.index, res["s"][str(mod)], label="MF")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("s")
    ax.set_title(f"Module {mod}")
    ax.legend()
plt.tight_layout()

fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
ax = axes[0]
ax.imshow(cov_snn, interpolation="none", aspect="equal")
ax.set_xlabel("module id")
ax.set_ylabel("module id")
ax.set_title("SNN")
ax = axes[1]
ax.imshow(cov_mf, interpolation="none", aspect="equal")
ax.set_xlabel("module id")
ax.set_ylabel("module id")
ax.set_title("MF")
plt.tight_layout()

plt.show()

