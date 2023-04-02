from rectipy import Network, circular_connectivity
import sys
cond, wdir, tdir = sys.argv[-3:]
sys.path.append(wdir)
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

# model parameters
N = 2000
p = 0.2
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
Delta = 0.1
eta = 30.0
a = 0.03
b = -2.0
d = 10.0
g = 15.0
E_r = 0.0
tau_s = 6.0
v_spike = 1000.0
v_reset = -1000.0

# device for computations
device = "cuda:0"

# working directory
# wdir = "config"
# tdir = "results"

# sweep condition
# cond = 165
p1 = "Delta"
p2 = "trial"

# parameter sweep definition
with open(f"{wdir}/bump_sweep.pkl", "rb") as f:
    sweep = pickle.load(f)
    v1s = sweep[p1]
    v2s = sweep[p2]
    f.close()
vals = [(v1, v2) for v1 in v1s for v2 in v2s]
v1, v2 = vals[int(cond)]
print(f"Condition: {p1} = {v1},  {p2} = {v2}")

# input parameters
cutoff = 500.0
T = 3000.0
dt = 1e-2
sr = 10
p_in = 0.1
n_inputs = int(N*p_in)
start = int(0.3*N)
stop = int(0.6*N)
margin = int(0.01*N)
distances = np.arange(start+n_inputs+margin, stop, step=margin)

# time-averaging parameters
sigma = 200
window = [27500, 29500]

# adjust parameters according to sweep condition
for param, v in zip([p1, p2], [v1, v2]):
    exec(f"{param} = {v}")

# define lorentzian of etas
thetas = lorentzian(N, eta=v_t, delta=Delta, lb=v_r, ub=0.0)

# define connectivity
indices = np.arange(0, N, dtype=np.int32)
pdfs = np.asarray([dist(idx, method="inverse", zero_val=0.0, inverse_pow=1.5) for idx in indices])
pdfs /= np.sum(pdfs)
W = circular_connectivity(N, p, spatial_distribution=rv_discrete(values=(indices, pdfs)), homogeneous_weights=False)

# simulation
############

# prepare results storage
results = {"sweep": {p1: v1, p2: v2}, "T": T, "dt": dt, "sr": sr, "p": p, "population_dists": [], "target_dists": [],
           "p_in": [], "W": W, "thetas": thetas}

for i, distance in enumerate(distances):

    # define inputs
    n_inputs = int(N * p_in)
    inp = np.zeros((int(T/dt), N))
    inp[:int(cutoff*0.5/dt), :] -= 30.0
    inp_indices = list(np.arange(start, start+n_inputs)) + list(np.arange(distance, distance+n_inputs))
    inp[int(1000/dt):int(1500/dt), inp_indices] += 30.0

    # initialize model
    node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": thetas, "eta": eta, "tau_u": 1/a, "b": b, "kappa": d,
                 "g": g, "E_r": E_r, "tau_s": tau_s, "v": v_t}

    # initialize model
    net = Network.from_yaml(f"{wdir}/ik_snn/rs", weights=W, source_var="s", target_var="s_in",
                            input_var="I_ext", output_var="s", spike_var="spike", spike_def="v", to_file=False,
                            node_vars=node_vars.copy(), op="rs_op", spike_reset=v_reset, spike_threshold=v_spike,
                            dt=dt, verbose=False, clear=True, device="cpu")

    # perform simulation
    obs = net.run(inputs=inp, sampling_steps=sr, record_output=True, verbose=False)
    res = obs["out"]

    # calculate the distribution of the time-averaged network activity after the stimulation was turned off
    s = gaussian_filter1d(res, sigma=200, axis=0)
    population_dist = np.mean(s[window[0]: window[1], :], axis=0).squeeze()
    population_dist /= np.sum(population_dist)
    target_dist = np.zeros((N,))
    target_dist[inp_indices] = 1.0/len(inp_indices)

    # store results
    results["target_dists"].append(target_dist)
    results["population_dists"].append(population_dist)

    # plot results
    # fig, axes = plt.subplots(nrows=2, figsize=(12, 8))
    # ax = axes[0]
    # im = ax.imshow(s.T, aspect=4.0, interpolation="none")
    # plt.colorbar(im, ax=ax, shrink=0.8)
    # ax.set_xlabel('time')
    # ax.set_ylabel('neurons')
    # ax = axes[1]
    # ax.plot(target_dist, label="target")
    # ax.plot(population_dist, label="SNN")
    # ax.set_xlabel("neurons")
    # ax.set_ylabel("probability")
    # plt.tight_layout()
    # plt.show()

# save results
fname = f"snn_multibump"
with open(f"{tdir}/{fname}_{cond}.pkl", "wb") as f:
    pickle.dump(results, f)
    f.close()
