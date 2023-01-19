from rectipy import Network, random_connectivity, input_connections
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


def sigmoid(x, kappa, t_on, omega):
    return 1.0/(1.0 + np.exp(-kappa*(x-np.cos(t_on*np.pi/omega))))


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
tau_s = 6.0
v_spike = 1000.0
v_reset = -1000.0

# parameter sweep definition
with open("config/impulse_response_sweep.pkl", "rb") as f:
    sweep = pickle.load(f)
    v1s = sweep[p1]
    v2s = sweep[p2]
    f.close()
vals = [(v1, v2) for v1 in v1s for v2 in v2s]
v1, v2 = vals[cond]

# input definition
T = 21000.0
dt = 1e-2
steps = int(T/dt)
sampling_steps = 100
freqs = [0.001]
m = len(freqs)
I_ext = np.zeros((steps, m))
for i, f in enumerate(freqs):
    I_ext[:, i] = sigmoid(np.cos(np.linspace(0, T, steps)*2.0*np.pi*f), kappa=2e3, t_on=1.0, omega=1.0/f)
alpha = 500.0

# simulation
############

results = {"s": [], "J": [], "I_ext": I_ext, "thetas": [], "sweep": {p1: v1, p2: v2}, "T": T, "dt": dt,
           "sr": sampling_steps, "W_in": []}
n_reps = 10
for idx in range(n_reps):

    # adjust parameters according to sweep condition
    for param, v in zip([p1, p2], [v1, v2]):
        exec(f"{param} = {v}")

    # create connectivity matrices
    J = random_connectivity(N, N, p, normalize=True)
    W_in = input_connections(N, m, 1.0, variance=1.0, zero_mean=True)

    # create background current distribution
    thetas = lorentzian(N, v_t, Delta, v_r, 2 * v_t - v_r)

    # collect remaining model parameters
    node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": thetas, "eta": eta, "tau_u": 1 / a, "b": b, "kappa": d, "g": g,
                 "E_r": E_r, "tau_s": tau_s, "v": v_t}

    # initialize model
    net = Network.from_yaml("neuron_model_templates.spiking_neurons.ik.ik", weights=J, source_var="s", target_var="s_in",
                            input_var="I_ext", output_var="s", spike_var="spike", spike_def="v", to_file=False,
                            node_vars=node_vars.copy(), op="ik_op", spike_reset=v_reset, spike_threshold=v_spike, dt=dt,
                            verbose=False, clear=True)
    net.add_input_layer(m, W_in, trainable=False)

    # simulation
    obs = net.run(inputs=I_ext * alpha, sampling_steps=sampling_steps, record_output=True, verbose=False)

    # results storage
    results["s"].append(obs["out"])
    results["J"].append(J)
    results["thetas"].append(thetas)
    results["W_in"].append(W_in)

    print(f"Run {idx} done for condition {cond}.")

# save results
# file name for saving
fname = f"ir_{p1}_{p2}"
with open(f"/projects/p31302/richard/results/{fname}_{cond}.pkl", "wb") as f:
    pickle.dump(results, f)
    f.close()
