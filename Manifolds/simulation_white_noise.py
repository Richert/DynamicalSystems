import pandas as pd
from rectipy import Network, random_connectivity, input_connections
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import cauchy
from scipy.ndimage import gaussian_filter1d


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

# file name for saving
fname = "wn_delta_5"

# network parameters
N = 1000
p = 0.1
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
Delta = 1.0
eta = 40.0
a = 0.03
b = -2.0
d = 300.0
g = 8.0
E_r = 0.0
tau_r = 2.0
tau_d = 8.0
v_spike = 1000.0
v_reset = -1000.0

# collect remaining model parameters
node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": [], "eta": eta, "tau_u": 1/a, "b": b, "kappa": d, "g": g,
             "E_r": E_r, "tau_r": tau_r, "tau_d": tau_d, "v": v_t}

# simulation settings
T = 21000.0
dt = 1e-1
steps = int(T/dt)
sampling_steps = 10

# input definition
p_in = 0.2
alpha = 200.0
sigma = 40
stimuli = np.random.randn(steps, 1)

# generate input
I_ext = np.zeros_like(stimuli)
I_ext[:, 0] = gaussian_filter1d(input=stimuli[:, 0], sigma=sigma)
I_ext[:, 0] /= np.max(np.abs(I_ext[:, 0]))

# parameter sweep definition
params = ["p"]
values = [[0.03125], [0.0625], [0.125], [0.25], [0.5], [1.0]]

# simulation
############

results = []
correlations = []
thetas = []
W_ins = []
Js = []
for vs in values:

    # change parameters
    for param, v in zip(params, vs):
        if param in node_vars:
            node_vars[param] = v
        else:
            exec(f"{param} = {v}")

    # draw random variables
    J = random_connectivity(N, N, p, normalize=True)
    W_in = input_connections(N, stimuli.shape[1], p_in, variance=1.0, zero_mean=True)
    node_vars["v_theta"] = lorentzian(N, v_t, Delta, v_r, 2 * v_t - v_r)

    # initialize model
    net = Network.from_yaml("neuron_model_templates.spiking_neurons.ik.ik_biexp", weights=J, source_var="s",
                            target_var="s_in", input_var="I_ext", output_var="s", spike_var="spike", spike_def="v",
                            node_vars=node_vars.copy(), op="ik_biexp_op", spike_reset=v_reset, spike_threshold=v_spike,
                            dt=dt, device="cuda:0")
    net.add_input_layer(stimuli.shape[1], W_in, trainable=False)

    # simulation
    obs = net.run(inputs=I_ext * alpha, sampling_steps=sampling_steps, record_output=True)
    results.append(obs["out"])

    # results storage
    projections = np.sum(J > 0, axis=0)
    correlations.append(np.corrcoef(W_in[:, 0], projections))
    Js.append(J)
    W_ins.append(W_in)
    thetas.append(node_vars["v_theta"])

# save results
inp = pd.DataFrame(index=results[-1].index, data=I_ext[::sampling_steps, :], columns=np.arange(0, stimuli.shape[1]))
pickle.dump({"s": results, "J": Js, "heterogeneity": thetas, "I_ext": inp, "W_in": W_ins, "params": node_vars,
             "T": T, "dt": dt, "sr": sampling_steps, "v_reset": v_reset, "v_spike": v_spike, "sweep": (params, values)},
            open(f"results/{fname}.pkl", "wb"))

# exemplary plotting
for vs, s, c in zip(values, results, correlations):
    _, ax = plt.subplots(figsize=(10, 4))
    ax.plot(s.mean(axis=1), color="blue")
    ax2 = ax.twinx()
    ax2.plot(inp.iloc[:, 0], color="orange")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("s")
    ax2.set_ylabel("I")
    condition = ", ".join([f"{param} = {v}" for param, v in zip(params, vs)])
    plt.title(f"{condition} : corr = {c[0, 1]}")
    plt.show()
