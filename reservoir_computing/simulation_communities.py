import numpy as np
import matplotlib.pyplot as plt
from pyrates import NodeTemplate, CircuitTemplate
from rectipy import Network
from networkx import to_pandas_adjacency
from utility_funcs import lorentzian, get_module_coupling, fit_lorentzian, community_coupling, get_community_input
from scipy.ndimage import gaussian_filter1d

# model parameters
n_comms = 5
n_neurons = 200
N = int(n_comms*n_neurons)
p_in = 0.1
p_out = 0.05
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
cutoff = 100.0
T = 1000.0 + cutoff
dt = 1e-2
sr = 100
alpha = 0.05
sigma = 100.0
steps = int(T/dt)
cutoff_steps = int(cutoff/(dt*sr))
I_ext = np.zeros((steps, N))
I_mf = np.zeros((steps, n_comms))
for i in range(n_comms):
    inp = gaussian_filter1d(np.random.randn(steps, 1), sigma=sigma)*alpha
    I_ext[:, i*n_neurons:(i+1)*n_neurons] = inp
    I_mf[:, i] = inp[:, 0]

# meta parameters
device = "cuda:0"

# simulate SNN dynamics
#######################

# create SNN connectivity
W = community_coupling(p_in, p_out, n_comms, n_neurons, sigma=0.02)
print(np.sum(np.sum(W, axis=1)))

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
    modules[str(idx)] = [i+counter for i in range(n_neurons)]
    counter += n_neurons

# get mean-field connectivity
W_mf = get_module_coupling(W, modules)
print(np.sum(np.sum(W_mf, axis=1)))

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
mf_res = mf.run(simulation_time=T, step_size=dt, sampling_step_size=dt * sr, outputs={"s": "all/rs_mf_op/s"},
                inputs={f"{n}/rs_mf_op/s_ext": I_mf[:, i] for i, n in enumerate(nodes)},
                solver="euler", cutoff=cutoff)

# calculate module covariance patterns
mf_var = np.mean([np.var(mf_res["s"].iloc[:, i]) for i in range(n_comms)])
snn_var = np.mean([np.var(snn_res.iloc[:, i]) for i in range(n_comms)])
mf_res_noise = mf_res["s"].values.T
# mf_res_noise += np.random.randn(*mf_res_noise.shape)*(snn_var-mf_var)
cov_mf = np.corrcoef(mf_res_noise)
cov_snn = np.corrcoef(np.asarray([np.mean(snn_res.iloc[:, idx]) for idx in modules.values()]))

# plotting
fig, axes = plt.subplots(nrows=len(modules), figsize=(10, 2*len(modules)))
for i, mod in enumerate(modules):
    ax = axes[i]
    ax.plot(mf_res.index, np.mean(snn_res.iloc[:, modules[mod]], axis=1), label="SNN")
    ax.plot(mf_res.index, mf_res_noise[i, :], label="MF")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("s")
    ax.set_title(f"Module {i}")
    ax.legend()
plt.tight_layout()

fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 7))
ax = axes[0, 0]
im = ax.imshow(cov_snn, interpolation="none", aspect="equal", vmax=1.0, vmin=0.0)
ax.set_xlabel("module id")
ax.set_ylabel("module id")
ax.set_title("Corr (SNN)")
plt.colorbar(im, ax=ax, shrink=0.8)
ax = axes[0, 1]
im = ax.imshow(cov_mf, interpolation="none", aspect="equal", vmax=1.0, vmin=0.0)
ax.set_xlabel("module id")
ax.set_ylabel("module id")
ax.set_title("Corr (MF)")
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
ax = axes[1, 0]
im = ax.imshow(W, interpolation="none", aspect="equal")
ax.set_xlabel("module id")
ax.set_ylabel("module id")
ax.set_title("W (SNN)")
plt.colorbar(im, ax=ax, shrink=0.8)
ax = axes[1, 1]
im = ax.imshow(W_mf, interpolation="none", aspect="equal")
ax.set_xlabel("module id")
ax.set_ylabel("module id")
ax.set_title("W (MF)")
plt.colorbar(im, ax=ax, shrink=0.8)
plt.show()

