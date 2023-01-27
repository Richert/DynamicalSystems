from rectipy import Network, random_connectivity
import numpy as np
import pickle
from scipy.stats import cauchy
import sys


def lorentzian(n: int, eta: float, delta: float, lb: float, ub: float):
    samples = np.zeros((n,))
    for i in range(n):
        s = cauchy.rvs(loc=eta, scale=delta)
        while s <= lb or s >= ub:
            s = cauchy.rvs(loc=eta, scale=delta)
        samples[i] = s
    return samples


# model definition
##################

# sweep condition
cond = int(sys.argv[-3])
p1 = str(sys.argv[-2])
p2 = str(sys.argv[-1])

# network parameters
N = 1000
p = 0.05
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
Delta = 1.0
eta = 55.0
a = 0.03
b = -2.0
d = 100.0
g = 10.0
E_r = 0.0
tau_r = 2.0
tau_d = 8.0
v_spike = 1000.0
v_reset = -1000.0

# parameter sweep definition
with open("/home/rgf3807/PycharmProjects/DynamicalSystems/Manifolds/config/sweep_autonomous_dynamics.pkl", "rb") as f:
    sweep = pickle.load(f)
    v1s = sweep[p1]
    v2s = sweep[p2]
    f.close()
vals = [(v1, v2) for v1 in v1s for v2 in v2s]
v1, v2 = vals[cond]

# simulation settings
T = 11000.0
dt = 1e-1
steps = int(T/dt)
sampling_steps = 10

# input definition
I_ext = np.zeros((steps, 1))

# simulation
############

results = {"s": [], "J": [], "thetas": [], "sweep": {p1: v1, p2: v2}, "T": T, "dt": dt, "sr": sampling_steps}
n_reps = 10
for idx in range(n_reps):

    # adjust parameters according to sweep condition
    for param, v in zip([p1, p2], [v1, v2]):
        exec(f"{param} = {v}")

    # create connectivity matrices
    J = random_connectivity(N, N, p, normalize=True)

    # create background current distribution
    thetas = lorentzian(N, v_t, Delta, v_r, 2 * v_t - v_r)

    # collect remaining model parameters
    node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": thetas, "eta": eta, "tau_u": 1 / a, "b": b, "kappa": d, "g": g,
                 "E_r": E_r, "tau_r": tau_r, "tau_d": tau_d, "v": v_t}

    # initialize model
    net = Network.from_yaml("neuron_model_templates.spiking_neurons.ik.ik_biexp", weights=J, source_var="s",
                            target_var="s_in", input_var="I_ext", output_var="s", spike_var="spike", spike_def="v",
                            to_file=False, node_vars=node_vars.copy(), op="ik_biexp_op", spike_reset=v_reset,
                            spike_threshold=v_spike, dt=dt, verbose=False, clear=True)

    # simulation
    obs = net.run(inputs=I_ext, sampling_steps=sampling_steps, record_output=True, verbose=False)

    # results storage
    results["s"].append(obs["out"])
    results["J"].append(J)
    results["thetas"].append(thetas)

    print(f"Run {idx} done for condition {cond}.")

# save results
# file name for saving
fname = f"ir_{p1}_{p2}"
with open(f"/projects/p31302/richard/results/{fname}_{cond}.pkl", "wb") as f:
    pickle.dump(results, f)
    f.close()
