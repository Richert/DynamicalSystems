import pandas as pd
from rectipy import Network, random_connectivity, input_connections
import numpy as np
import matplotlib.pyplot as plt
import pickle
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
fname = "ir_tanh_data"

# network parameters
N = 1000
p = 0.05
tau = 10.0
k = 1.0
eta = 0.0
Delta = 2.0
etas = eta + Delta*np.tan((np.pi/2)*(2.*np.arange(1, N+1)-N-1)/(N+1))

# create connectivity matrix
J = random_connectivity(N, N, p, normalize=True)

# collect remaining model parameters
node_vars = {"tau": tau, "eta": etas, "k": k}

# input definition
T = 11000.0
dt = 1e-2
steps = int(T/dt)
sampling_steps = 100
freqs = [0.001]
m = len(freqs)
alpha = 20.0
I_ext = np.zeros((steps, m))
for i, f in enumerate(freqs):
    I_ext[:, i] = sigmoid(np.cos(np.linspace(0, T, steps)*2.0*np.pi*f), kappa=2000, t_on=1.0, omega=1.0/f)
W_in = input_connections(N, m, 1.0, variance=1.0, zero_mean=True)
plt.plot(I_ext)
plt.show()

# parameter sweep definition
params = ["alpha", "k"]
values = [[100.0, 10.0], [100.0, 20.0], [200.0, 10.0], [200.0, 20.0]]

# simulation
############

results = []
for vs in values:

    for param, v in zip(params, vs):
        if param == "p":
            J = random_connectivity(N, N, p, normalize=True)
        elif param == "alpha":
            alpha = v
        else:
            node_vars[param] = v

    # initialize model
    net = Network.from_yaml("neuron_model_templates.rate_neurons.leaky_integrator.tanh", weights=J,
                            source_var="tanh_op/r", target_var="r_in", input_var="I_ext", output_var="v",
                            node_vars=node_vars.copy(), op="li_op", dt=dt, device="cuda:0")
    net.add_input_layer(m, W_in, trainable=False)

    # simulation
    obs = net.run(inputs=I_ext * alpha, sampling_steps=sampling_steps, record_output=True)
    results.append(obs["out"])

# save results
inp = pd.DataFrame(index=results[-1].index, data=I_ext[::sampling_steps, :], columns=np.arange(0, m))
pickle.dump({"s": results, "J": J, "I_ext": inp, "W_in": W_in, "params": node_vars, "T": T,
             "dt": dt, "sr": sampling_steps, "sweep": (params, values)},
            open(f"results/{fname}.pkl", "wb"))

# exemplary plotting
for vs, s in zip(values, results):
    _, ax = plt.subplots()
    ax.plot(s.mean(axis=1), color="blue")
    ax2 = ax.twinx()
    ax2.plot(inp.iloc[:, 0], color="orange")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("s")
    ax2.set_ylabel("I")
    plt.title(",".join([f"{param} = {v}" for param, v in zip(params, vs)]))
    plt.show()
