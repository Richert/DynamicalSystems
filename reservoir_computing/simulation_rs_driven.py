import matplotlib.pyplot as plt
import numpy as np
from pyrates import CircuitTemplate, NodeTemplate
from scipy.signal import welch

# network definition
####################

# define network nodes
ko = NodeTemplate.from_yaml("model_templates.oscillators.kuramoto.sin_pop")
ik = NodeTemplate.from_yaml("model_templates.neural_mass_models.ik.ik_theta_pop")
nodes = {'ik': ik, 'ko': ko}

# define network edges
alpha = 0.0005
edges = [
    ('ko/sin_op/s', 'ik/ik_theta_op/r_in', None, {'weight': alpha}),
    ('ik/ik_theta_op/r', 'ik/ik_theta_op/r_in', None, {'weight': 1.0})
]

# initialize network
net = CircuitTemplate(name="ik_forced", nodes=nodes, edges=edges)

# update izhikevich parameters
node_vars = {
    "C": 100.0,
    "k": 0.7,
    "v_r": -60.0,
    "v_t": -40.0,
    #"eta": 55.0,
    "Delta": 1.0,
    "g": 15.0,
    "E_r": 0.0,
    "b": -2.0,
    "a": 0.03,
    "d": 100.0,
    "tau_s": 6.0,
}
node, op = "ik", "ik_theta_op"
net.update_var(node_vars={f"{node}/{op}/{var}": val for var, val in node_vars.items()})

# update kuramoto parameters
omega = 0.001
net.update_var(node_vars={"ko/phase_op/omega": omega})

# perform simulation
####################

# simulation parameters
cutoff = 0.0
T = 100000.0 + cutoff
dt = 1e-2
dts = 1e-1
inp = np.zeros((int(T/dt),)) + 55.0

# perform simulation
res = net.run(T, dt, sampling_step_size=dts, cutoff=cutoff,
              inputs={"ik/ik_theta_op/I_ext": inp},
              outputs={"ko": "ko/phase_op/theta", "ik": "ik/ik_theta_op/r"},
              solver="euler", float_precision="float64")

# save results
res.to_csv("results/rs_driven_hom.csv")

# calculate psd of firing rate dynamics
freqs, pows = welch(res["ik"].values.squeeze(), fs=1/dts, window="hamming", nperseg=8192)
entrained_freq = freqs[np.argmax(pows)]

# plot results
fig, axes = plt.subplots(nrows=3, figsize=(12, 8))
ax = axes[0]
ax.plot(res.index, res["ik"]*1e3)
ax.set_xlabel("time (ms)")
ax.set_ylabel("r (Hz)")
ax = axes[1]
ax.plot(res.index, np.sin(2.0*np.pi*res["ko"]))
ax.set_xlabel("time (ms)")
ax.set_ylabel("input")
ax = axes[2]
ax.plot(freqs[freqs < 0.1]*1e3, pows[freqs < 0.1])
ax.set_xlabel("f (Hz)")
ax.set_ylabel("PSD")
ax.set_title(f"Entrained freq = {entrained_freq} for omega = {omega} and alpha = {alpha}")

plt.tight_layout()
plt.show()
