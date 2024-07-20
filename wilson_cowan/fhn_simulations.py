from pyrates import CircuitTemplate
import numpy as np
import matplotlib.pyplot as plt


# simulation parameters
T = 1000.0
start = 300.0
stop = 600.0
amp = 1.0
dt = 0.01
dts = 0.1

# model parameters
alpha = 0.05
a = 1.0
b = 0.5
node_vars = {"alpha": alpha, "a": a, "b": b}

# define input
inp = np.zeros((int(T/dt),)) + 1.0
inp[int(start/dt):int(stop/dt)] -= amp

# initialize model
fhn = CircuitTemplate.from_yaml("fhn/fhn")
fhn.update_var(node_vars={f"p/fhn_op/{key}": val for key, val in node_vars.items()})

# perform simulation
res = fhn.run(simulation_time=T, step_size=dt, sampling_step_size=dts, backend="default", solver="scipy",
              inputs={"p/fhn_op/I_ext": inp}, outputs={"v": "p/fhn_op/v", "u": "p/fhn_op/u"})

# plot results
res.plot()
plt.show()
