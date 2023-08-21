import numpy as np
import matplotlib.pyplot as plt
from pycobi import ODESystem
from pyrates import CircuitTemplate


# get a steady-state solution to start from
###########################################

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
              in_place=False, vectorize=False, rtol=1e-7, atol=1e-7)

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
_, _, params, state_vars = net.get_run_func(func_name="wc_rhs", file_name="wc", step_size=dt, auto=True,
                                            backend="fortran", solver='scipy', vectorize=False,
                                            float_precision='float64', in_place=False)

# perform bifurcation analysis
##############################

# initialize pycobi instance
ode = ODESystem(eq_file="wc", auto_dir="~/PycharmProjects/auto-07p", init_cont=False, params=params[3:],
                state_vars=list(state_vars))

# define the standard auto parameters
algorithm_params = {"NTST": 400, "NCOL": 4, "IAD": 3, "IPLT": 0, "NBC": 0, "NINT": 0, "NMX": 8000, "NPR": 5,
                    "MXBF": 5, "IID": 2, "ITMX": 40, "ITNW": 40, "NWTN": 12, "JAC": 0, "EPSL": 1e-7, "EPSU": 1e-7,
                    "EPSS": 1e-7, "DS": 1e-3, "DSMIN": 1e-10, "DSMAX": 2e-2, "IADS": 1, "THL": {}, "THU": {}}

# perform a 1D parameter continuation in S_e
ode.run(bidirectional=True, name="s_e:ss", c="ivp", ICP="E/wc_e/s", IPS=1, ILP=1, ISP=2, ISW=1, RL0=-1.0, RL1=8,
        **algorithm_params, UZR={"E/wc_e/s": 1.0})

# continue the limit cycle solution that must exist at the Hopf bifurcation found in the previous continuation
ode.run(starting_point="HB1", origin="s_e:ss", name="s_e:lc", ISW=-1, IPS=2, ISP=2, STOP={"BP1", "LP2"})

# change values of a and e
ode.run(starting_point="UZ1", origin="s_e:ss", name="a:ss", ICP=3, IPS=1, ILP=1, ISP=2, ISW=1, RL0=0.0, RL1=40,
        UZR={3: 36}, STOP=["UZ1"])
ode.run(starting_point="UZ1", origin="a:ss", name="e:ss", ICP=4, IPS=1, ILP=1, ISP=2, ISW=1, RL0=-60.0, RL1=0, DS="-",
        UZR={4: -55}, STOP=["UZ1"])

# perform second 1D parameter continuation in S_e
ode.run(name="s_e:ss:2", starting_point="UZ1", origin="e:ss", ICP="E/wc_e/s", IPS=1, ILP=1, ISP=2,
        ISW=1, RL0=-1.0, RL1=8.0, bidirectional=True)
ode.run(starting_point="HB1", origin="s_e:ss:2", name="s_e:lc:2", ISW=-1, IPS=2, ISP=2, STOP={"BP1", "LP2"})

# plot the 1D bifurcation diagram for the standard values
fig, axes = plt.subplots(nrows=2, figsize=(12, 6))
ax = axes[0]
ode.plot_continuation("E/wc_e/s", "E/wc_e/u", cont="s_e:ss", ax=ax, color="blue")
ode.plot_continuation("E/wc_e/s", "E/wc_e/u", cont="s_e:lc", ax=ax, color="black", ignore=["BP"])
ax.set_title("Equilibria in E as a function of S_e")
ax.set_xlabel("S_e")
ax.set_ylabel("E")
ax = axes[1]
ode.plot_continuation("E/wc_e/s", "I/wc_i/u", cont="s_e:ss", ax=ax, color="orange")
ode.plot_continuation("E/wc_e/s", "I/wc_i/u", cont="s_e:lc", ax=ax, color="black", ignore=["BP"])
ax.set_title("Equilibria in I as a function of S_e")
ax.set_xlabel("S_e")
ax.set_ylabel("I")
plt.suptitle("a = 16, e = 15")
plt.tight_layout()
fig.canvas.draw()
plt.savefig("wc_bifurcation_diagram1.pdf")

# plot the 1D bifurcation diagram for the standard values
fig, axes = plt.subplots(nrows=2, figsize=(12, 6))
ax = axes[0]
ode.plot_continuation("E/wc_e/s", "E/wc_e/u", cont="s_e:ss:2", ax=ax, color="blue")
ode.plot_continuation("E/wc_e/s", "E/wc_e/u", cont="s_e:lc:2", ax=ax, color="black", ignore=["BP"])
ax.set_title("Equilibria in E as a function of S_e")
ax.set_xlabel("S_e")
ax.set_ylabel("E")
ax = axes[1]
ode.plot_continuation("E/wc_e/s", "I/wc_i/u", cont="s_e:ss:2", ax=ax, color="orange")
ode.plot_continuation("E/wc_e/s", "I/wc_i/u", cont="s_e:lc:2", ax=ax, color="black", ignore=["BP"])
ax.set_title("Equilibria in I as a function of S_e")
ax.set_xlabel("S_e")
ax.set_ylabel("I")
plt.suptitle("a = 36, e = 55")
plt.tight_layout()
fig.canvas.draw()
plt.savefig("wc_bifurcation_diagram2.pdf")

plt.show()
