import sys
cond, wdir, tdir, device = sys.argv[-4:]
sys.path.append(wdir)
from rectipy import Network, random_connectivity, circular_connectivity
from pyrecu import modularity, sort_via_modules
import numpy as np
import pickle
from scipy.stats import rv_discrete
from utility_funcs import lorentzian, dist, get_dim
from pandas import DataFrame


# parameters and preparations
#############################

# working directory
# wdir = "config"
# tdir = "results"

# sweep condition
# cond = 0
p1 = "p"
p2 = "Delta"

# network parameters
N = 1000
p = 0.1
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
Delta = 1.0
etas = [50.0, 55.0]
a = 0.03
b = -2.0
ds = [10.0, 100.0]
g = 15.0
E_r = 0.0
tau_s = 6.0
v_spike = 1000.0
v_reset = -1000.0
connectivity = "random"

# parameter sweep definition
with open(f"{wdir}/dimensionality_sweep.pkl", "rb") as f:
    sweep = pickle.load(f)
    v1s = sweep[p1]
    v2s = sweep[p2]
    f.close()
vals = [(v1, v2) for v1 in v1s for v2 in v2s]
v1, v2 = vals[int(cond)]

# simulation parameters
cutoff = 2000.0
T = 3000.0 + cutoff
dt = 1e-2
sr = 100
steps = int(np.round(T/dt))
cutoff_steps = int(np.round(cutoff/(dt*sr)))
time = np.linspace(0.0, T, num=steps)
fs = int(np.round(1e3/(dt*sr), decimals=0))

# extrinsic input
I_ext = np.zeros((steps, 1))

# simulation
############

# prepare results storage
results = {"sweep": {p1: v1, p2: v2}, "T": T, "dt": dt, "sr": sr, "ds": ds, "p": p}
n_reps = 5
res_cols = ["d", "dim"]
dimensionalities = DataFrame(np.zeros((n_reps, len(res_cols))), columns=res_cols)
modules = {"d": [], "s": [], "m": [], "cov": [], "W": []}

# loop over repetitions
i = 0
for d, eta in zip(ds, etas):
    for _ in range(n_reps):

        # simulation preparations
        #########################

        # adjust parameters according to sweep condition
        for param, v in zip([p1, p2], [v1, v2]):
            exec(f"{param} = {v}")

        # create connectivity matrix
        if connectivity == "circular":
            indices = np.arange(0, N, dtype=np.int32)
            pdfs = np.asarray([dist(idx, method="inverse_squared") for idx in indices])
            pdfs /= np.sum(pdfs)
            W = circular_connectivity(N, p, spatial_distribution=rv_discrete(values=(indices, pdfs)))
        else:
            W = random_connectivity(N, N, p, normalize=True)

        # create background current distribution
        thetas = lorentzian(N, v_t, Delta, v_r, 2 * v_t - v_r)

        # collect remaining model parameters
        node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": thetas, "eta": eta, "tau_u": 1/a, "b": b, "kappa": d,
                     "g": g, "E_r": E_r, "tau_s": tau_s, "v": v_t}

        # initialize model
        net = Network.from_yaml(f"{wdir}/ik/rs", weights=W, source_var="s", target_var="s_in",
                                input_var="s_ext", output_var="s", spike_var="spike", spike_def="v", to_file=False,
                                node_vars=node_vars.copy(), op="rs_op", spike_reset=v_reset, spike_threshold=v_spike,
                                dt=dt, verbose=False, clear=True, device=device)

        # simulation
        ############

        obs = net.run(inputs=I_ext, sampling_steps=sr, record_output=True, verbose=False)
        rs = obs["out"].iloc[cutoff_steps:, :]

        # postprocessing
        ########################

        # calculate dimensionality of network dynamics
        dim, cov = get_dim(rs.values)

        # calculate modularity
        m, adj, nodes = modularity(cov, threshold=0.1, min_connections=5, min_nodes=50, decorator=None)
        cov = sort_via_modules(adj, m)
        signals = {"time": rs.index}
        for key, (indices, _) in m.items():
            signals[key] = np.mean(rs.values[:, nodes[indices]], axis=1)

        # save results
        dimensionalities.loc[i, "dim"] = dim
        dimensionalities.loc[i, "d"] = d
        modules["d"].append(d)
        modules["m"].append(m)
        modules["cov"].append(cov)
        modules["s"].append(signals)
        modules["W"].append(W)

        # go to next run
        i += 1
        print(f"Run {i} done for condition {cond}.")

# save results
fname = f"rs_dimensionality"
results["dim"] = dimensionalities
results["modules"] = modules
with open(f"{tdir}/{fname}_{cond}.pkl", "wb") as f:
    pickle.dump(results, f)
    f.close()
