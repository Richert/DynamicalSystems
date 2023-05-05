import numpy as np
import matplotlib.pyplot as plt
from pyrates import grid_search, CircuitTemplate
import pickle
from scipy.signal import find_peaks


# define model
net = CircuitTemplate.from_yaml("config/ik_mf/ik")
node_vars = {
    "C": 100.0,
    "k": 0.7,
    "v_r": -60.0,
    "v_t": -40.0,
    "eta": 55.0,
    "g": 15.0,
    "E_r": 0.0,
    "b": -2.0,
    "a": 0.03,
    "d": 100.0,
    "tau_s": 6.0,
}
net.update_var(node_vars={f"p/ik_op/{var}": val for var, val in node_vars.items()})

# load data that maps deltas to frequencies
data = pickle.load(open("results/fre_oscillations.pkl", "rb"))
deltas = data["deltas"]
freqs = data["freqs"]
param_map = {"Delta": {"vars": [f"ik_op/Delta"], "nodes": ['p']}}

# simulation parameters
cutoff = 2000.0
T = 10000.0 + cutoff
dt = 1e-2
dts = 1.0
inp = np.zeros((int(T/dt),))

# perform sweep
res, res_map = grid_search(net, param_grid={"Delta": deltas}, param_map=param_map, simulation_time=T, step_size=dt,
                           sampling_step_size=dts, outputs={"r": "p/ik_op/r"}, solver="scipy", method="RK45",
                           cutoff=cutoff)