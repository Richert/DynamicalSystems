import matplotlib.pyplot as plt
import numpy as np
from pyrates import CircuitTemplate, NodeTemplate

# network definition
####################

# define network nodes
ko = NodeTemplate.from_yaml("model_templates.oscillators.kuramoto.sin_pop")
ik = NodeTemplate.from_yaml("model_templates.neural_mass_models.ik.ik_theta_pop")
nodes = {'ik': ik, 'ko': ko}

# define network edges
edges = [
    ('ko/sin_op/s', 'ik/ik_theta_op/r_in', None, {'weight': 0.002}),
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
net.update_var(node_vars={"ko/phase_op/omega": 0.004})

# perform simulation
####################

# simulation parameters
T = 2000.0
dt = 1e-3
inp = np.zeros((int(T/dt),)) + 55.0

# perform simulation
from numba import njit
res = net.run(T, dt, inputs={"ik/ik_theta_op/I_ext": inp},
              outputs={"inp": "ko/phase_op/theta", "r": "ik/ik_theta_op/r"},
              solver="scipy", method="RK23", atol=1e-5, rtol=1e-4)

# save results
res.to_csv("results/rs_driven_hom.csv")

# plot results
fig, axes = plt.subplots(nrows=2, figsize=(12, 6))
ax = axes[0]
ax.plot(res.index, res["r"]*1e3)
ax.set_xlabel("time (ms)")
ax.set_ylabel("r (Hz)")
ax = axes[1]
ax.plot(res.index, np.sin(2.0*np.pi*res["inp"]))
ax.set_xlabel("time (ms)")
ax.set_ylabel("input")

plt.tight_layout()
plt.show()
