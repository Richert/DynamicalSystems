import pandas as pd
from rectipy import Network, random_connectivity, input_connections
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage import gaussian_filter1d
from scipy.stats import cauchy


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

# file name for saving
fname = "ir_qif_data"

# network parameters
N = 1000
p = 0.05
tau = 1.0
Delta = 0.2
eta = -0.1
J = 8.0
alpha = 0.3
tau_a = 10.0
tau_s = 0.6
v_spike = 1000.0
v_reset = -1000.0

# create connectivity matrix
W = random_connectivity(N, N, p, normalize=True)

# input definition
T = 600.0
dt = 1e-3
steps = int(T/dt)
sampling_steps = 100
freqs = [0.02]
m = len(freqs)
amp = 20.0 * Delta
I_ext = np.zeros((steps, m))
for i, f in enumerate(freqs):
    I_ext[:, i] = sigmoid(np.cos(np.linspace(0, T, steps)*2.0*np.pi*f), kappa=5000, t_on=0.1, omega=1.0/f)
W_in = input_connections(N, m, 0.1, variance=1.0, zero_mean=True)

# parameter sweep definition
param = "Delta"
values = np.asarray([0.2, 0.8, 2.0])

# simulation
############

results = []
for v in values:

    # create background current distribution
    etas = lorentzian(N, eta, Delta, eta - 100.0, eta + 100.0)

    # collect remaining model parameters
    node_vars = {"tau": tau, "eta": etas, "k": J*np.sqrt(Delta), "alpha": alpha, "tau_x": tau_a, "tau_s": tau_s}

    if param == "eta":
        node_vars["eta"] = v*Delta + Delta*np.tan((np.pi/2)*(2.*np.arange(1, N+1)-N-1)/(N+1))
    elif param == "Delta":
        amp *= v/Delta
        node_vars["k"] = J*np.sqrt(v)
        node_vars["eta"] = eta*v + v*np.tan((np.pi/2)*(2.*np.arange(1, N+1)-N-1)/(N+1))
    elif param == "p":
        W = random_connectivity(N, N, p, normalize=True)
    else:
        node_vars[param] = v

    # initialize model
    net = Network.from_yaml("neuron_model_templates.spiking_neurons.qif.qif_sfa", weights=W, source_var="s",
                            target_var="s_in", input_var="I_ext", output_var="s", spike_var="spike", spike_def="v",
                            node_vars=node_vars.copy(), op="qif_sfa_op", spike_reset=v_reset, spike_threshold=v_spike,
                            dt=dt)
    net.add_input_layer(m, W_in, trainable=False)

    # simulation
    obs = net.run(inputs=I_ext*amp, device="cpu", sampling_steps=sampling_steps, record_output=True)
    results.append(obs["out"])

    # plotting
    # s = obs["v"]
    # _, ax = plt.subplots()
    # for idx in np.argwhere(np.abs(W_in[:, 0]) > 1.0).squeeze():
    #     ax.plot(s.iloc[:, idx])
    # plt.show()

# save results
inp = pd.DataFrame(index=results[-1].index, data=I_ext[::sampling_steps, :], columns=np.arange(0, m))
pickle.dump({"s": results, "J": J, "heterogeneity": etas, "I_ext": inp, "W_in": W_in, "params": node_vars,
             "dt": dt, "sr": sampling_steps, "v_reset": v_reset, "v_spike": v_spike, "sweep": (param, values)},
            open(f"results/{fname}.pkl", "wb"))

# exemplary plotting
for v, s in zip(values, results):
    _, ax = plt.subplots()
    ax.plot(s.mean(axis=1), color="blue")
    plt.title(f"{param} = {v}")
    plt.show()
