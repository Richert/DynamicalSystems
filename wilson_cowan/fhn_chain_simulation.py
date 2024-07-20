from pyrates import NodeTemplate, CircuitTemplate
import numpy as np
import matplotlib.pyplot as plt


# simulation parameters
T = 3000.0
start = 300.0
stop = 600.0
amp = 0.0
dt = 0.01
dts = 0.1

# model parameters
N = 10
alpha = 0.02
a = 1.0
b = 0.5
c = 1.0
node_vars = {"alpha": alpha, "a": a, "b": b}

# define input
inp = np.zeros((int(T/dt),)) - 2.0
inp[int(start/dt):int(stop/dt)] -= amp

# initialize model
fhn = NodeTemplate.from_yaml("fhn/fhn_pop")
net = CircuitTemplate("net", nodes={f"p{i}": fhn for i in range(N)},
                      edges=[(f"p{i}/fhn_op/v", f"p{i+1}/fhn_op/v_ext", None, {"weight": c}) for i in range(N-1)])
net.update_var(node_vars={f"all/fhn_op/{key}": val for key, val in node_vars.items()})

# perform simulation
res = net.run(simulation_time=T, step_size=dt, sampling_step_size=dts, backend="default", solver="scipy",
              inputs={"all/fhn_op/I_ext": inp}, outputs={"v": "all/fhn_op/v"})

# plot results
res.plot()
plt.show()
