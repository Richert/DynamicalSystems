from pyrates import CircuitTemplate, NodeTemplate
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os


def fit_lorentzian(x: np.ndarray, modules: dict, nodes: list) -> np.ndarray:
    deltas = []
    x = x[nodes]
    for _, (indices, ) in modules.items():
        x_mod = x[indices]
        deltas.append(np.std(x_mod))
    return np.asarray(deltas)


def get_module_coupling(W: np.ndarray, modules: dict, nodes: list) -> np.ndarray:
    W = W[nodes, :]
    W = W[:, nodes]
    W_mod = np.zeros((len(modules), len(modules)))
    for i, mod1 in enumerate(modules):
        targets, _ = modules[mod1]
        for j, mod2 in enumerate(modules):
            sources, _ = modules[mod1]
            W_tmp = W[targets, :]
            W_tmp = W_tmp[:, sources]
            W_mod[i, j] = np.mean(W_tmp > 0.0)
    return W_mod


# load SNN data
module_examples = {"lc": [], "ss": []}
example = 0
condition = {"Delta": 1.0, "p": 0.03125}
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
                                             "W": mods["W"][i], "nodes": mods[i]["nodes"], "thetas": mods[i]["thetas"]}
                                            )
                example += 1

# extract condition
idx = 0
cond = "ss"
example = module_examples[cond][idx]
W = example["W"]
thetas = example["thetas"]
modules = example["m"]
nodes = example["nodes"]

# fit theta distribution for each module
deltas = fit_lorentzian(thetas, modules, nodes)

# approximate the connectivity within and between modules
W_mods = get_module_coupling(W, modules, nodes)

# create mean-field model
rs = NodeTemplate.from_yaml("results/ik/rs_mf")
mf = CircuitTemplate("rs", nodes={key: rs for key in modules})
mf.add_edges_from_matrix("p/rs_mf_op/s", "p/rs_mf_op/s_in", nodes=list(modules.keys()), weight=W_mods)

# simulate mean-field model dynamics
cutoff = 2000.0
T = 3000.0 + cutoff
dt = 1e-3
sr = 1000
res = mf.run(simulation_time=T, step_size=dt, sampling_step_size=dt*sr, outputs={"s": "p/rs_mf_op/s"})

# plotting
fig, axes = plt.subplots(nrows=len(modules), figsize=(10, len(modules)))
for i, mod in enumerate(modules):
    ax = axes[i]
    ax.plot(res.index, example["s"][mod], label="SNN")
    ax.plot(res.index, res["s"][mod], label="MF")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("s")
    ax.set_title(f"Module {mod}")
    plt.legend()
plt.tight_layout()
plt.show()
