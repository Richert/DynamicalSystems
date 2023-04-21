import torch.cuda
from rectipy import Network, circular_connectivity
from pyrates import NodeTemplate, CircuitTemplate
import sys
# cond, wdir, tdir = sys.argv[-3:]
# sys.path.append(wdir)
sys.path.append("~/PycharmProjects/DynamicalSystems/reservoir_computing")
import numpy as np
from scipy.stats import cauchy
import matplotlib.pyplot as plt
import pickle
from scipy.stats import rv_discrete
from scipy.ndimage import gaussian_filter1d


def lorentzian(n: int, eta: float, delta: float, lb: float, ub: float):
    samples = np.zeros((n,))
    for i in range(n):
        s = cauchy.rvs(loc=eta, scale=delta)
        while s <= lb or s >= ub:
            s = cauchy.rvs(loc=eta, scale=delta)
        samples[i] = s
    return samples


def dist(x: int, method: str = "inverse", zero_val: float = 1.0, inverse_pow: float = 1.0) -> float:
    if method == "inverse":
        return 1/x**inverse_pow if x > 0 else zero_val
    if method == "exp":
        return np.exp(-x) if x > 0 else zero_val
    else:
        raise ValueError("Invalid method.")


# define parameters
###################

# device for computations
device = "cuda:0"

# working directory
wdir = "config"
tdir = "results"

# sweep condition
cond = 12
p1 = "Delta_i"
p2 = "trial"

# parameter sweep definition
with open(f"{wdir}/eic_sweep.pkl", "rb") as f:
    sweep = pickle.load(f)
    v1s = sweep[p1]
    v2s = sweep[p2]
    f.close()
vals = [(v1, v2) for v1 in v1s for v2 in v2s]
v1, v2 = vals[int(cond)]
print(f"Condition: {p1} = {v1},  {p2} = {v2}")

# general parameters
N = 1000
p = 0.2
v_spike = 1e3
v_reset = -1e3

# RS neuron parameters
Ce = 100.0   # unit: pF
ke = 0.7  # unit: None
ve_r = -60.0  # unit: mV
ve_t = -40.0  # unit: mV
Delta_e = 0.1  # unit: mV
de = 10.0
ae = 0.03
be = -2.0
Ie = 60.0

# FS neuron parameters
Ci = 20.0   # unit: pF
ki = 1.0  # unit: None
vi_r = -55.0  # unit: mV
vi_t = -40.0  # unit: mV
Delta_i = 0.2  # unit: mV
di = 0.0
ai = 0.2
bi = 0.025
Ii = 45.0

# synaptic parameters
g_ampa = 1.0
g_gaba = 1.0
E_ampa = 0.0
E_gaba = -65.0
tau_ampa = 6.0
tau_gaba = 8.0
k_ee = 15.0
k_ei = 15.0
k_ie = 8.0
k_ii = 8.0

# input parameters
cutoff = 500.0
T = 3000.0
dt = 1e-2
sr = 10
p_in_vals = 10.0**np.linspace(-1.0, -0.1, num=2)
inp = torch.zeros((int(T/dt), 2*N), device=device)

# time-averaging parameters
sigma = 200
window = [27500, 29500]

# adjust parameters according to sweep condition
for param, v in zip([p1, p2], [v1, v2]):
    exec(f"{param} = {v}")

# define lorentzian of etas
spike_thresholds_e = lorentzian(N, eta=ve_t, delta=Delta_e, lb=ve_r, ub=2*ve_t - ve_r)
spike_thresholds_i = lorentzian(N, eta=vi_t, delta=Delta_i, lb=vi_r, ub=2*vi_t - vi_r)

# define connectivity
indices = np.arange(0, N, dtype=np.int32)
e_pdfs = np.asarray([dist(idx, method="inverse", zero_val=0.0, inverse_pow=1.5) for idx in indices])
e_pdfs /= np.sum(e_pdfs)
i_pdfs = np.asarray([dist(idx, method="inverse", zero_val=0.0, inverse_pow=1.5) for idx in indices])
i_pdfs /= np.sum(i_pdfs)
W_ee = circular_connectivity(N, p, spatial_distribution=rv_discrete(values=(indices, e_pdfs)), homogeneous_weights=False)
W_ie = circular_connectivity(N, p, spatial_distribution=rv_discrete(values=(indices, e_pdfs)), homogeneous_weights=False)
W_ei = circular_connectivity(N, p, spatial_distribution=rv_discrete(values=(indices, i_pdfs)), homogeneous_weights=False)
W_ii = circular_connectivity(N, p, spatial_distribution=rv_discrete(values=(indices, i_pdfs)), homogeneous_weights=False)

# initialize nodes
e_vars = {"C": Ce, "k": ke, "v_r": ve_r, "v_theta": spike_thresholds_e, "eta": Ie, "tau_u": 1 / ae, "b": be,
          "kappa": de,
          "g_e": g_ampa, "E_e": E_ampa, "g_i": g_gaba, "E_i": E_gaba, "tau_s": tau_ampa, "v": ve_t}
i_vars = {"C": Ci, "k": ki, "v_r": vi_r, "v_theta": spike_thresholds_i, "eta": Ii, "tau_u": 1 / ai, "b": bi,
          "kappa": di,
          "g_e": g_ampa, "E_e": E_ampa, "g_i": g_gaba, "E_i": E_gaba, "tau_s": tau_gaba, "v": vi_t}
rs = NodeTemplate.from_yaml("config/ik_snn/ik")
fs = NodeTemplate.from_yaml("config/ik_snn/ik")

# construct rs and fs circuits
rs_neurons = {f'rs_{i}': rs for i in range(N)}
rs_net = CircuitTemplate("rs", nodes=rs_neurons)
rs_net.add_edges_from_matrix(source_var="ik_op/s", target_var="ik_op/s_e", weight=W_ee * k_ee,
                             source_nodes=list(rs_neurons.keys()))
rs_net.update_var(node_vars={f"all/ik_op/{key}": val for key, val in e_vars.items()})
fs_neurons = {f'fs_{i}': fs for i in range(N)}
fs_net = CircuitTemplate("fs", nodes=fs_neurons)
fs_net.add_edges_from_matrix(source_var="ik_op/s", target_var="ik_op/s_i", weight=W_ii * k_ii,
                             source_nodes=list(fs_neurons.keys()))
fs_net.update_var(node_vars={f"all/ik_op/{key}": val for key, val in i_vars.items()})

# combine the rs and fs populations into a single circuit
net = CircuitTemplate("eic", circuits={"rs": rs_net, "fs": fs_net})
net.add_edges_from_matrix(source_var="ik_op/s", target_var="ik_op/s_e", weight=W_ie * k_ie,
                          source_nodes=[f"rs/rs_{i}" for i in range(N)],
                          target_nodes=[f"fs/fs_{i}" for i in range(N)])
net.add_edges_from_matrix(source_var="ik_op/s", target_var="ik_op/s_i", weight=W_ei * k_ei,
                          source_nodes=[f"fs/fs_{i}" for i in range(N)],
                          target_nodes=[f"rs/rs_{i}" for i in range(N)])

# initialize rectipy model
model = Network(dt=dt, device="cuda:0")
model.add_diffeq_node("eic", node=net, input_var="ik_op/I_ext", output_var="rs/all/ik_op/s",
                      spike_var="ik_op/spike", spike_def="all/all/ik_op/v", spike_reset=v_reset,
                      spike_threshold=v_spike, verbose=False, clear=True, to_file=False)

# simulation
############

# prepare results storage
results = {"sweep": {p1: v1, p2: v2}, "T": T, "dt": dt, "sr": sr, "p": p, "population_dists": [], "target_dists": [],
           "p_in": [], "thetas_rs": spike_thresholds_e, "thetas_fs": spike_thresholds_i}

for i, p_in in enumerate(p_in_vals):

    # define inputs
    n_inputs = int(N * p_in)
    center = int(N*0.5)
    inp_indices = np.arange(center-int(0.5*n_inputs), center+int(0.5*n_inputs))
    inp[:, :] = 0
    inp[:int(cutoff*0.5/dt), :N] -= 30.0
    inp[int(1000/dt):int(1500/dt), inp_indices] += 30.0

    # perform simulation
    obs = model.run(inputs=inp, sampling_steps=sr, record_output=True, verbose=False, enable_grad=False)
    res = obs.to_numpy("out")

    # calculate the distribution of the time-averaged network activity after the stimulation was turned off
    s = gaussian_filter1d(res, sigma=200, axis=0)
    population_dist = np.mean(s[window[0]: window[1], :], axis=0).squeeze()
    population_dist /= np.sum(population_dist)
    target_dist = np.zeros((N,))
    target_dist[inp_indices] = 1.0/len(inp_indices)

    # store results
    results["target_dists"].append(target_dist)
    results["population_dists"].append(population_dist)
    results["p_in"].append(p_in)

    # reset network state
    model.reset()

    # plot results
    fig, axes = plt.subplots(nrows=2, figsize=(12, 8))
    ax = axes[0]
    im = ax.imshow(s.T, aspect=4.0, interpolation="none")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xlabel('time')
    ax.set_ylabel('neurons')
    ax = axes[1]
    ax.plot(target_dist, label="target")
    ax.plot(population_dist, label="SNN")
    ax.set_xlabel("neurons")
    ax.set_ylabel("probability")
    plt.tight_layout()
    plt.show()

# save results
fname = f"snn_bump"
with open(f"{tdir}/{fname}_{cond}.pkl", "wb") as f:
    pickle.dump(results, f)
    f.close()
