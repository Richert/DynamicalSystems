import pandas as pd
from rectipy import Network, random_connectivity, input_connections
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage import gaussian_filter1d

# model definition
##################

# file name for saving
fname = "rnn_data"

# network parameters
N = 1000
p = 0.1
k = 2.5
tau = 8.0
Delta = 0.5
eta = 0.0

# create connectivity matrix
J = random_connectivity(N, N, p, normalize=True)

# create background current distribution
etas = eta + Delta*np.tan((np.pi/2)*(2.*np.arange(1, N+1)-N-1)/(N+1))

# collect remaining model parameters
node_vars = {"k": k, "tau": tau, "eta": etas}

# input definition
T = 4000.0
dt = 1e-2
steps = int(T/dt)
sampling_steps = 100
in_start = int(1100.0/dt)
sigma, amp = 100.0, 100.0
I_ext = np.random.uniform(low=-1.0, high=1.0, size=(steps, 1)) * amp
I_ext[:, 0] = gaussian_filter1d(I_ext[:, 0], sigma=sigma, axis=0)
W_in = input_connections(N, 1, 0.5, variance=5.0, zero_mean=False)

# parameter sweep definition
param = "eta"
values = np.linspace(-5.0, 5.0, num=7)

# simulation
############

results = []
for v in values:

    if param == "eta":
        node_vars[param] = v + Delta*np.tan((np.pi/2)*(2.*np.arange(1, N+1)-N-1)/(N+1))
    else:
        node_vars[param] = v

    # initialize model
    net = Network.from_yaml("neuron_model_templates.rate_neurons.leaky_integrator.tanh", weights=J,
                            source_var="tanh_op/r", target_var="r_in", input_var="I_ext", output_var="v",
                            node_vars=node_vars.copy(), op="li_op", dt=dt)
    net.add_input_layer(1, W_in, trainable=False)

    # simulation
    obs = net.run(inputs=I_ext, device="cuda", sampling_steps=sampling_steps, record_output=True)
    r = np.tanh(obs["out"])
    results.append(r)

# save results
inp = pd.DataFrame(index=results[-1].index, data=I_ext[::sampling_steps, :])
pickle.dump({"s": results, "J": J, "etas": etas, "I_ext": inp, "W_in": W_in, "params": node_vars,
             "dt": dt, "sr": sampling_steps, "sweep": (param, values)},
            open(f"results/{fname}.pkl", "wb"))

# exemplary plotting
for idx in [0, 2, 4]:
    s = results[idx]
    _, ax = plt.subplots()
    ax.plot(s.mean(axis=1), color="blue")
    plt.show()
