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

# define sweep
deltas = np.linspace(0.1, 1.3, num=20)
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

# calculate oscillation frequencies
freqs = []
for delta in deltas:
    idx = np.argmin(np.abs(delta - res_map.loc[:, "Delta"].values))
    s = res.iloc[:, idx].values
    s /= np.max(s)
    peaks, _ = find_peaks(s, width=5, height=0.5)
    # _, ax = plt.subplots(figsize=(10, 4))
    # ax.plot(s)
    # for p in peaks:
    #     ax.axvline(x=p, linestyle="--", color="blue")
    # ax.set_title(f"Frequency = {len(peaks)/10.0}")
    # plt.show()
    freqs.append(len(peaks)*1e3/T)

# save data
fn = "config/fre_oscillations.pkl"
pickle.dump({"res": res, "map": res_map, "deltas": deltas, "freqs": freqs},
            open(fn, "wb"))
