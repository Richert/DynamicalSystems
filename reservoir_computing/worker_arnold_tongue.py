from rectipy import Network, random_connectivity, circular_connectivity
import numpy as np
import pickle
from scipy.stats import cauchy, rv_discrete
import sys


def lorentzian(n: int, eta: float, delta: float, lb: float, ub: float):
    samples = np.zeros((n,))
    for i in range(n):
        s = cauchy.rvs(loc=eta, scale=delta)
        while s <= lb or s >= ub:
            s = cauchy.rvs(loc=eta, scale=delta)
        samples[i] = s
    return samples


def dist(x: int, method: str = "inverse") -> float:
    if method == "inverse":
        return 1/x if x > 0 else 1
    if method == "inverse_squared":
        return 1/x**2 if x > 0 else 1
    if method == "exp":
        return np.exp(-x)
    else:
        raise ValueError("Invalid method.")


# model definition
##################

# working directory
wdir = sys.argv[-2]
tdir = sys.argv[-1]

# sweep condition
cond = int(sys.argv[-3])
p1 = "p_in"
p2 = "alpha"

# network parameters
N = 1000
p = 0.1
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
Delta = 1.0
eta = 55.0
a = 0.03
b = -2.0
d = 100.0
g = 15.0
E_r = 0.0
tau_s = 6.0
v_spike = 1000.0
v_reset = -1000.0

# parameter sweep definition
with open(f"{wdir}/arnold_tongue_sweep.pkl", "rb") as f:
    sweep = pickle.load(f)
    v1s = sweep[p1]
    v2s = sweep[p2]
    f.close()
vals = [(v1, v2) for v1 in v1s for v2 in v2s]
v1, v2 = vals[cond]

# simulation parameters
cutoff = 30000.0
T = 300000.0 + cutoff
dt = 1e-2
sr = 100
steps = int(np.round(T/dt))
time = np.linspace(0.0, T, num=steps)

# input definition
omega = 0.005
alpha = 1.0
p_in = 0.1
I_ext = np.zeros((steps, 1))
I_ext[:, 0] = np.sin(2.0*np.pi*omega*time)

# simulation
############

results = {"s": [], "J": [], "thetas": [], "W_in": [], "sweep": {p1: v1, p2: v2}, "T": T, "dt": dt,
           "sr": sr}
n_reps = 5
for idx in range(n_reps):

    # adjust parameters according to sweep condition
    for param, v in zip([p1, p2], [v1, v2]):
        exec(f"{param} = {v}")

    # create connectivity matrix
    connectivity = "exponential"
    indices = np.arange(0, N, dtype=np.int32)
    pdfs = np.asarray([dist(idx, method="inverse_squared") for idx in indices])
    pdfs /= np.sum(pdfs)
    if connectivity == "circular":
        W = circular_connectivity(N, p, spatial_distribution=rv_discrete(values=(indices, pdfs)))
    else:
        W = random_connectivity(N, N, p, normalize=True)

    # create input matrix
    W_in = np.zeros((N, 1))
    idx = np.random.choice(np.arange(N), size=int(N*p_in), replace=False)
    W_in[idx, 0] = alpha

    # create background current distribution
    thetas = lorentzian(N, v_t, Delta, v_r, 2 * v_t - v_r)

    # collect remaining model parameters
    node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": thetas, "eta": eta, "tau_u": 1/a, "b": b, "kappa": d, "g": g,
                 "E_r": E_r, "tau_s": tau_s, "v": v_t}

    # initialize model
    net = Network.from_yaml(f"{wdir}/ik/rs", weights=W, source_var="s", target_var="s_in",
                            input_var="I_ext", output_var="s", spike_var="spike", spike_def="v", to_file=False,
                            node_vars=node_vars.copy(), op="rs_op", spike_reset=v_reset, spike_threshold=v_spike, dt=dt,
                            verbose=False, clear=True, device="cuda:0")
    net.add_input_layer(1, weights=W_in)

    # simulation
    obs = net.run(inputs=I_ext, sampling_steps=sr, record_output=True, verbose=False)

    # results storage
    results["s"].append(obs["out"])
    results["J"].append(W)
    results["thetas"].append(thetas)
    results["W_in"].append(W_in)

    print(f"Run {idx} done for condition {cond}.")

# save results
fname = f"rs_arnold_tongue"
with open(f"{tdir}/{fname}_{cond}.pkl", "wb") as f:
    pickle.dump(results, f)
    f.close()
