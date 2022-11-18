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


# model definition
##################

# file name for saving
fname = "snn_data"

# network parameters
N = 1000
p = 0.1
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
Delta = 1.0
eta = 0.0
a = 0.03
b = 0.0
d = 10.0
g = 20.0
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
T = 4000.0
dt = 1e-2
steps = int(T/dt)
sampling_steps = 100
in_start = int(1100.0/dt)
sigma, amp = 100.0, 40.0
I_ext = np.random.uniform(low=0, high=1.0, size=(steps, 1)) * amp #np.zeros((steps, 1))
# I_ext[in_start] = amp
I_ext = gaussian_filter1d(I_ext, sigma=sigma, axis=0)
W_in = input_connections(N, 1, 0.2, variance=5.0, zero_mean=False)

# parameter sweep definition
param = "Delta"
values = np.linspace(0.1, 5.0, num=7)

# simulation
############

results = []
for v in values:

    if param == "v_theta":
        node_vars["v_theta"] = v + Delta*np.tan((np.pi/2)*(2.*np.arange(1, N+1)-N-1)/(N+1))
    elif param == "Delta":
        node_vars["v_theta"] = v_t + v*np.tan((np.pi/2)*(2.*np.arange(1, N+1)-N-1)/(N+1))
    elif param == "p":
        J = random_connectivity(N, N, p, normalize=True)
    else:
        node_vars[param] = v

    # initialize model
    net = Network.from_yaml("neuron_model_templates.spiking_neurons.ik.ik", weights=J, source_var="s", target_var="s_in",
                            input_var="I_ext", output_var="s", spike_var="spike", spike_def="v",
                            node_vars=node_vars.copy(), op="ik_op", spike_reset=v_reset, spike_threshold=v_spike, dt=dt)
    net.add_input_layer(1, W_in, trainable=False)

    # simulation
    obs = net.run(inputs=I_ext, device="cpu", sampling_steps=sampling_steps, record_output=True,
                  record_vars=[("v", False)])
    results.append(obs["out"])

    # plotting
    # s = obs["v"]
    # _, ax = plt.subplots()
    # for idx in np.argwhere(np.abs(W_in[:, 0]) > 1.0).squeeze():
    #     ax.plot(s.iloc[:, idx])
    # plt.show()

# save results
inp = pd.DataFrame(index=results[-1].index, data=I_ext[::sampling_steps, :])
pickle.dump({"s": results, "J": J, "heterogeneity": thetas, "I_ext": inp, "W_in": W_in, "params": node_vars,
             "dt": dt, "sr": sampling_steps, "v_reset": v_reset, "v_spike": v_spike, "sweep": (param, values)},
            open(f"results/{fname}.pkl", "wb"))

# exemplary plotting
for idx in [0, 2, 4]:
    s = results[idx]
    _, ax = plt.subplots()
    ax.plot(s.mean(axis=1), color="blue")
    plt.show()
