import numpy as np
from pyrates import CircuitTemplate, NodeTemplate, grid_search
import pickle
import sys


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
    "Delta": 0.1,
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
alphas = np.linspace(1, 10, num=10)*1e-4
omegas = np.linspace(1, 6, num=10)*1e-3
sweep = {"alpha": alphas, "omega": omegas}
param_map = {"alpha": {"vars": ["weight"], "edges": [('ko/sin_op/s', 'ik/ik_theta_op/r_in')]},
             "omega": {"vars": ["phase_op/omega"], "nodes": ["ko"]}}

# simulation parameters
cutoff = 30000.0
T = 300000.0 + cutoff
dt = 1e-2
dts = 1e-1
inp = np.zeros((int(T/dt),)) + 55.0

# perform sweep
res, res_map = grid_search(net, param_grid=sweep, param_map=param_map, simulation_time=T, step_size=dt,
                           sampling_step_size=dts, cutoff=cutoff, permute_grid=True, vectorize=True,
                           inputs={"ik/ik_theta_op/I_ext": inp},
                           outputs={"ik": "ik/ik_theta_op/r", "ko": "ko/phase_op/theta"},
                           solver="euler", float_precision="float64"
                           )

# save data
fn = sys.argv[-1]
pickle.dump({"res": res, "map": res_map, "alphas": alphas, "omegas": omegas},
            open(f"{fn}_hom.pkl", "wb"))
