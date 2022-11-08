from rectipy import Network
import numpy as np
import matplotlib.pyplot as plt
import pickle

# load data
data = pickle.load(open("results/snn_data.pkl", "rb"))

# input definition
T = 1000.0
dt = data["dt"]
steps = int(T/dt)
sampling_steps = data["sr"]
sigma = 2.0
I_ext = np.random.randn(steps, 1)*sigma*np.sqrt(dt)
W_in = data["W_in"]

# initialize model
net = Network.from_yaml("neuron_model_templates.spiking_neurons.ik.iku", weights=data["J"], source_var="s",
                        target_var="s_in", input_var="I_ext", output_var="s", spike_var="spike", spike_def="v",
                        node_vars=data["params"], op="iku_op", spike_reset=data["v_reset"],
                        spike_threshold=data["v_spike"], dt=dt)
net.add_input_layer(1, W_in, trainable=False)

# simulation
############

# simulation
obs = net.run(inputs=I_ext, device="cpu", sampling_steps=sampling_steps, record_output=True, record_vars=[("v", False)])
