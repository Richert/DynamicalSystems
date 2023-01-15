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
fname = "ir_rs_data3"

# network parameters
N = 1000
p = 0.05
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
Delta = 1.0
eta = 20.0
a = 0.03
b = -2.0
d = 100.0
g = 25.0
E_r = 0.0
tau_s = 6.0
v_spike = 1000.0
v_reset = -1000.0

# create connectivity matrix
J = random_connectivity(N, N, p, normalize=True)

# create background current distribution
thetas = lorentzian(N, v_t, Delta, v_r, 2*v_t-v_r)

# collect remaining model parameters
node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": thetas, "eta": eta, "tau_u": 1/a, "b": b, "kappa": d, "g": g,
             "E_r": E_r, "tau_s": tau_s, "v": v_t}

# input definition
T = 11000.0
dt = 1e-2
steps = int(T/dt)
sampling_steps = 100
freqs = [0.001]
m = len(freqs)
alpha = 200.0
I_ext = np.zeros((steps, m))
for i, f in enumerate(freqs):
    I_ext[:, i] = sigmoid(np.cos(np.linspace(0, T, steps)*2.0*np.pi*f), kappa=2000, t_on=1.0, omega=1.0/f)
W_in = input_connections(N, m, 1.0, variance=1.0, zero_mean=True)
#W_in = np.abs(W_in)
plt.plot(I_ext)
plt.show()

# parameter sweep definition
params = ["eta", "Delta"]
values = [[20.0, 2.0], [20.0, 1.0], [10.0, 0.2], [10.0, 2.0]]

# simulation
############

results = []
for vs in values:

    for param, v in zip(params, vs):
        if param == "v_theta":
            node_vars["v_theta"] = v + Delta*np.tan((np.pi/2)*(2.*np.arange(1, N+1)-N-1)/(N+1))
        elif param == "Delta":
            node_vars["v_theta"] = v_t + v*np.tan((np.pi/2)*(2.*np.arange(1, N+1)-N-1)/(N+1))
        elif param == "p":
            J = random_connectivity(N, N, p, normalize=True)
        elif param == "alpha":
            alpha = v
        else:
            node_vars[param] = v

    # initialize model
    net = Network.from_yaml("neuron_model_templates.spiking_neurons.ik.ik", weights=J, source_var="s", target_var="s_in",
                            input_var="I_ext", output_var="s", spike_var="spike", spike_def="v",
                            node_vars=node_vars.copy(), op="ik_op", spike_reset=v_reset, spike_threshold=v_spike, dt=dt,
                            device="cuda:0")
    net.add_input_layer(m, W_in, trainable=False)

    # simulation
    obs = net.run(inputs=I_ext * alpha, sampling_steps=sampling_steps, record_output=True,
                  record_vars=[("v", False)])
    results.append(obs["out"])

    # plotting
    # s = obs["v"]
    # _, ax = plt.subplots()
    # for idx in np.argwhere(np.abs(W_in[:, 0]) > 1.0).squeeze():
    #     ax.plot(s.iloc[:, idx])
    # plt.show()

# save results
inp = pd.DataFrame(index=results[-1].index, data=I_ext[::sampling_steps, :], columns=np.arange(0, m))
pickle.dump({"s": results, "J": J, "heterogeneity": thetas, "I_ext": inp, "W_in": W_in, "params": node_vars, "T": T,
             "dt": dt, "sr": sampling_steps, "v_reset": v_reset, "v_spike": v_spike, "sweep": (params, values)},
            open(f"results/{fname}.pkl", "wb"))

# exemplary plotting
for v, s in zip(values, results):
    _, ax = plt.subplots()
    ax.plot(s.mean(axis=1), color="blue")
    ax2 = ax.twinx()
    ax2.plot(inp.iloc[:, 0], color="orange")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("s")
    ax2.set_ylabel("I")
    plt.title(f"{param} = {v}")
    plt.show()
