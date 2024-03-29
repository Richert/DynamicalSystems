import numpy as np
from pyrates import CircuitTemplate, NodeTemplate, grid_search
import pickle
import sys


# network definition
####################

# define network nodes
node, op = "ik_pop", "ik_op"
ko = NodeTemplate.from_yaml("model_templates.oscillators.kuramoto.sin_pop")
rs = NodeTemplate.from_yaml(f"config/ik_mf/{node}")
nodes = {node: rs, 'ko': ko}

# define network edges
edges = [
    ('ko/sin_op/s', f'{node}/{op}/I_ext', None, {'weight': 1.0}),
    (f'{node}/{op}/r', f'{node}/{op}/r_in', None, {'weight': 1.0})
]

# initialize network
net = CircuitTemplate(name="rs_forced", nodes=nodes, edges=edges)

# update izhikevich parameters
node_vars = {
    "C": 100.0,
    "k": 0.7,
    "v_r": -60.0,
    "v_t": -40.0,
    "eta": 55.0,
    "Delta": 0.1,
    "g": 15.0,
    "E_r": 0.0,
    "b": -2.0,
    "a": 0.03,
    "d": 100.0,
    "tau_s": 6.0,
}
net.update_var(node_vars={f"{node}/{op}/{var}": val for var, val in node_vars.items()})

# perform parameter sweep
#########################

# define sweep
alphas = 10**(np.linspace(0, 1, num=20))
omegas = np.linspace(2.0, 5.9, num=20)*1e-3
sweep = {"alpha": alphas, "omega": omegas}
param_map = {"alpha": {"vars": ["weight"], "edges": [('ko/sin_op/s', f'{node}/{op}/I_ext')]},
             "omega": {"vars": ["phase_op/omega"], "nodes": ["ko"]}}

# simulation parameters
cutoff = 10000.0
T = 120000.0 + cutoff
dt = 1e-2
dts = 1.0

# perform sweep
res, res_map = grid_search(net, param_grid=sweep, param_map=param_map, simulation_time=T, step_size=dt,
                           sampling_step_size=dts, cutoff=cutoff, permute_grid=True, vectorize=True,
                           outputs={"rs": f"{node}/{op}/r", "ko": "ko/phase_op/theta"},
                           solver="euler", float_precision="float64"
                           )

# save data
fn = "results/fre_at_hom.pkl"  #sys.argv[-1]
pickle.dump({"res": res, "map": res_map, "alphas": alphas, "omegas": omegas},
            open(fn, "wb"))
