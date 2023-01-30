import pandas as pd
from rectipy import Network
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


# model definition
##################

# file name for loading/saving
fname = "rs_ir"

# load config
config = pickle.load(open(f"config/{fname}_config.pkl", "rb"))

# load model/simulation variables
node_vars = config["node_vars"]
I_ext = config["inp"]
W = config["W"]
W_in = config["W_in"]
params = config["additional_params"]
dt = config["dt"]
sr = config["sr"]

# run-specific parameters (for sweeps)
alpha = 10.0
sweep = {"alpha": alpha}

# simulation
############

# initialize model
net = Network.from_yaml("neuron_model_templates.spiking_neurons.ik.ik", weights=W, source_var="s", target_var="s_in",
                        input_var="I_ext", output_var="s", spike_var="spike", spike_def="v",
                        node_vars=node_vars.copy(), op="ik_op", spike_reset=params["v_reset"],
                        spike_threshold=params["v_spike"], dt=dt, device="cuda:0")
net.add_input_layer(W_in.shape[1], W_in, trainable=False)

# simulation
obs = net.run(inputs=I_ext * alpha, sampling_steps=sr, record_output=True)

# save results
res = obs["out"]
inp = pd.DataFrame(index=res.index, data=I_ext[::sr, :], columns=np.arange(0, W_in.shape[1]))
pickle.dump({"s": res, "I_ext": inp, "sweep": sweep},
            open(f"results/{fname}_results.pkl", "wb"))

# exemplary plotting
_, axes = plt.subplots(nrows=2)
ax = axes[0]
ax.plot(res.mean(axis=1), color="blue")
ax.set_xlabel("time (ms)")
ax.set_ylabel("s")
ax2 = axes[1]
ax2.plot(inp.iloc[:, 0], color="orange")
ax2.set_xlabel("time (ms)")
ax2.set_ylabel("I")
plt.show()
