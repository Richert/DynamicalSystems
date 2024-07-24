from pyrates import NodeTemplate, CircuitTemplate
import numpy as np
import matplotlib.pyplot as plt


# simulation parameters
T = 200.0
start = 100.0
stop = 200.0
amp = -0.5
dt = 0.01
dts = 0.1

# model parameters
N = 10
tau = 10.0
a = 0.8
b = 0.5
g = 0.9
node_vars = {"tau": tau, "a": a, "b": b, "g": g}

# define input
inp = np.zeros((int(T/dt),)) - 1.0
inp[int(start/dt):int(stop/dt)] += amp

# initialize model
fhn = NodeTemplate.from_yaml("fhn/fhn_pop")
net = CircuitTemplate("net", nodes={f"p{i}": fhn for i in range(N)},
                      edges=[(f"p{i}/fhn_op/v", f"p{i+1}/fhn_op/v_net", None, {"weight": 1.0}) for i in range(N-1)])
net.update_var(node_vars={f"all/fhn_op/{key}": val for key, val in node_vars.items()})

# perform simulation
res = net.run(simulation_time=T, step_size=dt, sampling_step_size=dts, backend="default", solver="euler",
              inputs={"all/fhn_op/I_ext": inp}, outputs={"v": "all/fhn_op/v"})

# plot results
res.plot()
plt.show()
