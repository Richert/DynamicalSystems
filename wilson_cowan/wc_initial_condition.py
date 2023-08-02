import numpy as np
import matplotlib.pyplot as plt
from pyrates import CircuitTemplate


# initialize pyrates model
net = CircuitTemplate.from_yaml("wc/wc")

# define extrinsic input for numerical simulation
dt = 1e-4
T = 100.0
steps = int(T/dt)
inp = np.zeros((steps,))

# perform numerical simulation to receive well-defined initial condition
res = net.run(simulation_time=T, step_size=dt, inputs={"E/wc_e/s": inp},
              outputs={"E": "E/wc_e/u", "I": "I/wc_i/u"}, solver="scipy", method="RK23",
              in_place=False, vectorize=False)

# plot results
fig, axes = plt.subplots(nrows=2, figsize=(12, 5))
ax = axes[0]
ax.plot(res["E"])
ax.set_xlabel("time")
ax.set_ylabel("E")
ax.set_title("Excitatory population dynamics")
ax = axes[1]
ax.plot(res["I"], color="orange")
ax.set_xlabel("time")
ax.set_ylabel("I")
ax.set_title("Inhibitory population dynamics")
plt.tight_layout()
fig.canvas.draw()
plt.savefig("wc_initial_condition.pdf")
plt.show()

# generate pycobi files with initial condition
net.get_run_func(func_name="wc_rhs", file_name="wc", step_size=dt, auto=True, backend="fortran",
                 solver='scipy', vectorize=False, float_precision='float64', in_place=False)
