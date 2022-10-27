from rectipy import Network, random_connectivity
import numpy as np
import matplotlib.pyplot as plt
import pickle
# model definition
##################

# network parameters
N = 1000
p = 0.05
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
Delta = 0.5
eta = 60.0
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
             "E_r": E_r, "tau_s": tau_s}

# initialize model
net = Network.from_yaml("neuron_model_templates.spiking_neurons.ik.ik", weights=J, source_var="s", target_var="s_in",
                        input_var="I_ext", output_var="s", spike_var="spike", spike_def="v", node_vars=node_vars,
                        op="ik_op", spike_reset=v_reset, spike_threshold=v_spike)

# simulation
############

# input definition
T = 70000.0
dt = 5e-2
steps = int(T/dt)
I_ext = np.zeros((steps, 1))

# simulation
obs = net.run(inputs=I_ext, device="cpu", sampling_steps=100, record_output=True)

# save results
pickle.dump({"res": obs["out"], "J": J, "etas": etas}, open("results/snn_autonomous.pkl", "wb"))

# exemplary plotting
s = obs["out"]
plt.plot(s.mean(axis=1))
plt.show()
