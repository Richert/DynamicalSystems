import numpy as np
import matplotlib.pyplot as plt
from pyrates import CircuitTemplate

# parameters
node_vars = {"tau": 6.7, "J": 59.0, "eta": -6.3, "Delta": 0.9, "tau_a": 999.0, "kappa": 0.023}
T = 4000.0
dt = 1e-2
I_ext = 0.0
I_start = 40.0
I_stop = 160.0

# model
net = CircuitTemplate.from_yaml("config/qif_mf/qif_ca")
net.update_var(node_vars={f"p/qif_ca_op/{key}": val for key, val in node_vars.items()})

# run simulation
inp = np.zeros((int(T/dt),))
inp[int(I_start/dt):int(I_stop/dt)] = I_ext
res = net.run(simulation_time=T, step_size=dt, inputs={"p/qif_ca_op/I_ext": inp},
              outputs={"r": f"p/qif_ca_op/r", "a": f"p/qif_ca_op/a"},
              solver="heun", clear=False, sampling_step_size=100*dt
              )

# plotting
fig = plt.figure(figsize=(16, 6))
grid = fig.add_gridspec(nrows=2, ncols=1)
ax1 = fig.add_subplot(grid[0, :])
ax1.plot(res.index, res["r"])
ax1.set_ylabel("r (Hz)")
ax = fig.add_subplot(grid[1, :])
ax.sharex(ax1)
ax.plot(res.index, res["a"])
ax.set_ylabel("a")
ax.set_xlabel("time")
plt.tight_layout()
plt.show()
