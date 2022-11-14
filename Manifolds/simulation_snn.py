import pandas as pd
from rectipy import Network, random_connectivity, input_connections
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage import gaussian_filter1d

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
Delta = 2.0
eta = 46.0
a = 0.03
b = -2.0
d = 100.0
g = 15.0
E_r = 0.0
tau_s = 6.0
v_spike = 1000.0
v_reset = -1000.0

# create connectivity matrix
J = random_connectivity(N, N, p, normalize=True)

# create background current distribution
etas = eta + Delta*np.tan((np.pi/2)*(2.*np.arange(1, N+1)-N-1)/(N+1))

# collect remaining model parameters
node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": v_t, "eta": etas, "tau_u": 1/a, "b": b, "kappa": d, "g": g,
             "E_r": E_r, "tau_s": tau_s, "v": v_t}

# input definition
T = 4000.0
dt = 1e-2
steps = int(T/dt)
sampling_steps = 100
in_start = int(1100.0/dt)
sigma, amp = 200.0, 100.0
I_ext = np.zeros((steps, 1))
I_ext[in_start] = amp
I_ext = gaussian_filter1d(I_ext, sigma=sigma, axis=0)
W_in = input_connections(N, 1, 0.2, variance=5.0, zero_mean=False)

# parameter sweep definition
param = "eta"
values = np.linspace(30.0, 70.0, num=5)

# simulation
############

results = []
for v in values:

    if param == "eta":
        node_vars[param] += v - eta
    else:
        node_vars[param] = v

    # initialize model
    net = Network.from_yaml("neuron_model_templates.spiking_neurons.ik.ik", weights=J, source_var="s", target_var="s_in",
                            input_var="I_ext", output_var="s", spike_var="spike", spike_def="v",
                            node_vars=node_vars.copy(), op="ik_op", spike_reset=v_reset, spike_threshold=v_spike, dt=dt)
    net.add_input_layer(1, W_in, trainable=False)

    # simulation
    obs = net.run(inputs=I_ext, device="cpu", sampling_steps=sampling_steps, record_output=True)
    results.append(obs["out"])

# save results
inp = pd.DataFrame(index=results[-1].index, data=I_ext[::sampling_steps, :])
pickle.dump({"s": results, "J": J, "etas": etas, "I_ext": inp, "W_in": W_in, "params": node_vars,
             "dt": dt, "sr": sampling_steps, "v_reset": v_reset, "v_spike": v_spike, "sweep": (param, values)},
            open(f"results/{fname}.pkl", "wb"))

# exemplary plotting
for idx in [0, 2, 4]:
    s = results[idx]
    _, ax = plt.subplots()
    ax.plot(s.mean(axis=1), color="blue")
    ax2 = ax.twinx()
    ax2.plot(I_ext, color="orange")
    plt.legend(["s", "I_ext"])
    plt.show()
