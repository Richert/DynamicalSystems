# import sys
# cond, wdir, tdir, device = sys.argv[-4:]
# sys.path.append(wdir)
from rectipy import Network, random_connectivity, circular_connectivity
from sklearn.decomposition import SparsePCA, DictionaryLearning
from networkx import navigable_small_world_graph, to_pandas_adjacency
import numpy as np
import pickle
from scipy.stats import rv_discrete
from utility_funcs import lorentzian, dist, get_dim
from pandas import DataFrame


# parameters and preparations
#############################

device = "cuda:0"

# working directory
wdir = "config"
tdir = "results"

# sweep condition
cond = 9
p1 = "p"
p2 = "Delta"

# network parameters
N = int(30*30)
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
connectivity = "small_world"

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
modules = {"d": [], "s": [], "m": [], "cov": [], "W": [], "thetas": []}

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
        elif connectivity == "small_world":
            G = navigable_small_world_graph(30, p=1, q=np.maximum(1, int(N*0.01)), r=2, dim=2)
            W = to_pandas_adjacency(G).values
            W /= np.sum(W, axis=1)
        else:
            W = random_connectivity(N, N, p, normalize=True)
        import matplotlib.pyplot as plt
        plt.imshow(W)
        plt.show()

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
        model = DictionaryLearning(n_components=5, fit_algorithm="cd", transform_algorithm="lasso_cd", alpha=10.0,
                                   transform_alpha=0.01, n_jobs=-1, positive_code=True, positive_dict=True)
        # model = SparsePCA(n_components=5, alpha=2.0, ridge_alpha=1.0)
        s = model.fit_transform(rs.values)
        factors = model.components_

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
        ax = axes[0]
        ax.plot(np.mean(s, axis=1), label="fit")
        ax.plot(np.mean(rs.values, axis=1), label="target")
        ax.legend()
        ax = axes[1]
        im = ax.imshow(factors, interpolation="none", aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.show()

        # save results
        dimensionalities.loc[i, "dim"] = dim
        dimensionalities.loc[i, "d"] = d
        modules["d"].append(d)
        modules["m"].append(factors)
        modules["cov"].append(cov)
        modules["s"].append(s)
        modules["W"].append(W)
        modules["thetas"].append(thetas)

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
