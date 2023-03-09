from pyrates import CircuitTemplate, NodeTemplate
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import minimize_scalar
from scipy.stats import norm, cauchy


def fit_lorentzian(x: np.ndarray, modules: dict, nodes: list) -> tuple:
    deltas = []
    mus = []
    x = x[nodes]
    for indices, _ in modules.values():
        x_mod = x[indices]
        mu, delta = cauchy.fit(x_mod)
        deltas.append(delta)
        mus.append(mu)
    return np.asarray(deltas), np.asarray(mus)


def lorentzian(n: int, eta: float, delta: float, lb: float, ub: float):
    samples = np.zeros((n,))
    for i in range(n):
        s = cauchy.rvs(loc=eta, scale=delta)
        while s <= lb or s >= ub:
            s = cauchy.rvs(loc=eta, scale=delta)
        samples[i] = s
    return samples


def get_module_coupling(W: np.ndarray, modules: dict, nodes: list = None) -> np.ndarray:
    if nodes:
        W = W[nodes, :]
        W = W[:, nodes]
    W_mod = np.zeros((len(modules), len(modules)))
    for i, mod1 in enumerate(modules):
        targets, _ = modules[mod1]
        for j, mod2 in enumerate(modules):
            sources, _ = modules[mod2]
            W_tmp = W[targets, :]
            W_tmp = W_tmp[:, sources]
            W_mod[i, j] = np.mean(np.sum(W_tmp, axis=1))
    W_mod /= np.sum(W_mod, axis=1, keepdims=False)
    return W_mod


def gaussian(n, mu: float, sd: float, lb: float, ub: float):
    samples = np.zeros((n,))
    for i in range(n):
        s = norm.rvs(loc=mu, scale=sd)
        while s <= lb or s >= ub:
            s = norm.rvs(loc=mu, scale=sd)
        samples[i] = s
    return samples


def sample_sd_cauchy(delta: float, epsilon: float, targets: np.ndarray, mu: float, n_samples: int, lb: float, ub: float,
                     alpha: float = 0.99):
    samples = lorentzian(n_samples, eta=mu, delta=delta, lb=lb, ub=ub)
    # plt.hist(samples, label="fit")
    # plt.hist(targets, label="target")
    # plt.legend()
    # plt.show()
    epsilon = alpha*epsilon + (1-alpha)*(np.std(samples) - np.std(targets))**2
    return epsilon


# load SNN data
module_examples = {"lc": [], "ss": []}
example = 0
condition = {"Delta": 1.0, "p": 0.0625}
path = "results/dimensionality2"
fn = "rs_dimensionality"
for file in os.listdir(path):
    if fn in file:
        f = pickle.load(open(f"{path}/{file}", "rb"))
        dim = f["dim"]
        mods = f["modules"]
        p = f["sweep"]["p"]
        Delta = f["sweep"]["Delta"]
        for i in range(dim.shape[0]):
            dim_tmp = dim.iloc[i, :]
            include_example = True
            condition_test = {"Delta": Delta, "p": p}
            for key in condition:
                if np.abs(condition[key] - condition_test[key]) > 1e-4:
                    include_example = False
            if include_example:
                key = "ss" if dim_tmp["d"] < 50.0 else "lc"
                module_examples[key].append({"m": mods["m"][i], "s": mods["s"][i], "cov": mods["cov"][i],
                                             "W": mods["W"][i], "nodes": mods["nodes"][i], "thetas": mods["thetas"][i]}
                                            )
                example += 1

# extract condition
idx = 2
cond = "lc"
example = module_examples[cond][idx]
W = example["W"]
thetas = example["thetas"]
modules = example["m"]
nodes = example["nodes"]

# fit theta distribution for each module
deltas, thresholds = fit_lorentzian(thetas, modules, nodes)
print(f"Spike threshold distribution community means: {thresholds}")
print(f"Spike threshold distribution community widths: {deltas}")

# approximate the connectivity within and between modules
W_mods = get_module_coupling(W, modules, nodes)

# create mean-field model
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
Delta = condition["Delta"]
eta = 55.0
a = 0.03
b = -2.0
g = 15.0
E_r = 0.0
tau_s = 6.0
node_vars = {"C": C, "k": k, "v_r": v_r, "v_t": thresholds, "Delta": deltas, "tau_u": 1/a, "b": b, "E_r": E_r,
             "eta": eta, "kappa": 10.0 if condition == "ss" else 100.0, "g": g, "tau_s": tau_s, "v": v_t}
rs = NodeTemplate.from_yaml("config/ik/rs_mf")
nodes = [str(key) for key in modules]
mf = CircuitTemplate("rs", nodes={key: rs for key in nodes})
mf.add_edges_from_matrix("rs_mf_op/s", "rs_mf_op/s_in", nodes=nodes, weight=W_mods)
mf.update_var(node_vars={f"all/rs_mf_op/{key}": val for key, val in node_vars.items()})

# simulate mean-field model dynamics
cutoff = 2000.0
T = 3000.0 + cutoff
dt = 1e-3
sr = 1000
res = mf.run(simulation_time=T, step_size=dt, sampling_step_size=dt*sr, outputs={"s": "all/rs_mf_op/s"},
             solver="scipy", method="DOP853", cutoff=cutoff)

# calculate module covariance patterns
cov_mf = np.cov(res["s"].values.T)
cov_snn = np.cov(np.asarray([example["s"][idx] for idx in modules]))

# plotting
fig, axes = plt.subplots(nrows=len(modules), figsize=(10, 2*len(modules)))
for i, mod in enumerate(modules):
    ax = axes[i]
    ax.plot(res.index, example["s"][mod], label="SNN")
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
