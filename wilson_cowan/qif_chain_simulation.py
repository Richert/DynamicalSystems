from pyrates import CircuitTemplate, NodeTemplate
import numpy as np
import matplotlib.pyplot as plt

# simulation parameters
cutoff = 50.0
T = 800.0 + cutoff
start = 400.0 + cutoff
amp1 = -5.21
amp2 = -4.5
dt = 0.001
dts = 0.01

# model parameters
N = 10
alpha = 0.08
Delta = 2.0
eta = 0.0
J = 15*np.sqrt(Delta)
tau = 1.0
tau_a = 10.0
k = 0.05

node_vars = {"alpha": alpha, "tau": tau, "tau_a": tau_a, "Delta": Delta, "eta": eta, "J": J}

# define input
inp = np.zeros((int(T/dt),)) + amp1
inp[int(start/dt):] = amp2

# initialize model
# initialize model
qif = NodeTemplate.from_yaml("qif/qif_sd_pop")
net = CircuitTemplate("net", nodes={f"p{i}": qif for i in range(N)},
                      edges=[(f"p{i}/qif_sd_op/r", f"p{i+1}/qif_sd_op/r_in", None, {"weight": k}) for i in range(N-1)])
net.update_var(node_vars={f"all/qif_sd_op/{key}": val for key, val in node_vars.items()})

# perform simulation
res = net.run(simulation_time=T, step_size=dt, sampling_step_size=dts, backend="default", solver="euler",
              inputs={"all/qif_sd_op/I_ext": inp}, outputs={"r": "all/qif_sd_op/r"}, cutoff=cutoff)

# plot results
fig, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(res["r"].T, aspect="auto", interpolation="none", cmap="viridis")
plt.colorbar(im, ax=ax, shrink=0.7)
ax.set_xlabel("steps")
ax.set_ylabel("neurons")
plt.tight_layout()
plt.show()
