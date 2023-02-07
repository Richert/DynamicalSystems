import numpy as np
from pyrates import CircuitTemplate, NodeTemplate, grid_search
from numba import njit
import pickle


# network definition
####################

# define network nodes
ko = NodeTemplate.from_yaml("model_templates.oscillators.kuramoto.sin_pop")
ik = NodeTemplate.from_yaml("model_templates.neural_mass_models.ik.ik_theta_pop")
nodes = {'ik': ik, 'ko': ko}

# define network edges
edges = [
    ('ko/sin_op/s', 'ik/ik_theta_op/r_in', None, {'weight': 0.001}),
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

# perform parameter sweep
#########################

# define sweep
alphas = np.asarray([0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064])
omegas = np.linspace(-0.003, 0.003, 11) + 0.004
sweep = {"alpha": alphas, "omega": omegas}
param_map = {"alpha": {"vars": ["weight"], "edges": [('ko/sin_op/s', 'ik/ik_theta_op/r_in')]},
             "omega": {"vars": ["phase_op/omega"], "nodes": ["ko"]}}

# simulation parameters
T = 11000.0
dt = 1e-3
dts = 1e-2
inp = np.zeros((int(T/dt),)) + 55.0
cutoff = 1000.0

# perform sweep
res, res_map = grid_search(net, param_grid=sweep, param_map=param_map, simulation_time=T, step_size=dt,
                           solver="scipy", method="RK23", atol=1e-5, rtol=1e-4, sampling_step_size=dts,
                           permute_grid=True, vectorize=True, inputs={"ik/ik_theta_op/I_ext": inp},
                           outputs={"ik": "ik/ik_theta_op/r", "ko": "ko/phase_op/theta"})

# save data
pickle.dump({"res": res, "map": res_map, "alphas": alphas, "omegas": omegas},
            open("results/rs_arnold_tongue.pkl", "wb"))
