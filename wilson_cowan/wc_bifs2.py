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

# change edge weights
net.update_var(edge_vars=[("E/wc_e/u", "E/wc_e/u_in", {"weight": 36.0}),
                          ("I/wc_i/u", "E/wc_e/u_in", {"weight": -55.0})])

# perform numerical simulation to receive well-defined initial condition
res = net.run(simulation_time=T, step_size=dt, inputs={"E/wc_e/s": inp},
              outputs={"E": "E/wc_e/u", "I": "I/wc_i/u"}, solver="scipy", method="RK23",
              in_place=False, vectorize=False, rtol=1e-7, atol=1e-7)

# plot results
# fig, axes = plt.subplots(nrows=2, figsize=(12, 5))
# ax = axes[0]
# ax.plot(res["E"])
# ax.set_xlabel("time")
# ax.set_ylabel("E")
# ax.set_title("Excitatory population dynamics")
# ax = axes[1]
# ax.plot(res["I"], color="orange")
# ax.set_xlabel("time")
# ax.set_ylabel("I")
# ax.set_title("Inhibitory population dynamics")
# plt.tight_layout()
# fig.canvas.draw()
# plt.savefig("wc_initial_condition.pdf")
# plt.show()

# generate pycobi files with initial condition
_, _, params, state_vars = net.get_run_func(func_name="wc_rhs", file_name="wc", step_size=dt, auto=True,
                                            backend="fortran", solver='scipy', vectorize=False,
                                            float_precision='float64', in_place=False)

# initialize pycobi instance
ode = ODESystem(eq_file="wc", auto_dir="~/PycharmProjects/auto-07p", init_cont=False, params=params[3:],
                state_vars=list(state_vars))

# define the standard auto parameters
algorithm_params = {"NTST": 400, "NCOL": 4, "IAD": 3, "IPLT": 0, "NBC": 0, "NINT": 0, "NMX": 8000, "NPR": 5,
                    "MXBF": 7, "IID": 2, "ITMX": 40, "ITNW": 40, "NWTN": 12, "JAC": 0, "EPSL": 1e-8, "EPSU": 1e-8,
                    "EPSS": 1e-9, "DS": 1e-3, "DSMIN": 1e-12, "DSMAX": 5e-2, "IADS": 1, "THL": {}, "THU": {}}

# perform bifurcation analysis for S_i = 1.35
#############################################

# perform a 1D parameter continuation in S_e
ode.run(bidirectional=True, name="s_e:ss", c="ivp", ICP="E/wc_e/s", IPS=1, ILP=1, ISP=2, ISW=1, RL0=-8.0, RL1=20.0,
        **algorithm_params, UZR={"E/wc_e/s": 1.0})

# continue the limit cycle solution that must exist at the Hopf bifurcation found in the previous continuation
ode.run(starting_point="HB1", origin="s_e:ss", name="s_e:lc", ISW=-1, IPS=2, ISP=2, STOP={"BP1", "LP2"})

# perform bifurcation analysis for S_i = 0.0
############################################

# change value of S_i for isolated WC oscillator
ode.run(starting_point="UZ1", origin="s_e:ss", name="s_i:ss", ICP="I/wc_i/s", IPS=1, ILP=1, ISP=2, ISW=1, RL0=-2.0,
        RL1=4.0, bidirectional=True, UZR={"I/wc_i/s": 0.0})

# perform second 1D parameter continuation in S_e
ode.run(name="s_e:ss:2", starting_point="UZ1", origin="s_i:ss", ICP="E/wc_e/s", IPS=1, ILP=1, ISP=2,
        ISW=1, RL0=-8.0, RL1=20.0, bidirectional=True, STOP=[], UZR={})
ode.run(starting_point="HB1", origin="s_e:ss:2", name="s_e:lc:2", ISW=-1, IPS=2, ISP=2, STOP={"BP1", "LP3"},
        reduce_limit_cycle=True)

# 2D bifurcation diagram in S_i and S_e
########################################

ode.run(starting_point="HB1", origin="s_e:ss", name="s_e/s_i:hb1", ICP=["I/wc_i/s", "E/wc_e/s"],
        IPS=1, ISP=2, ILP=0, ISW=2, IPLT=9, RL0=-2.0, RL1=4.0, bidirectional=True)
ode.run(starting_point="HB1", origin="s_e:ss:2", name="s_e/s_i:hb2", ICP=["I/wc_i/s", "E/wc_e/s"],
        IPS=1, ISP=2, ILP=0, ISW=2, IPLT=9, RL0=-2.0, RL1=4.0, bidirectional=True)
ode.run(starting_point="LP2", origin="s_e:ss", name="s_e/s_i:lp1", ICP=["I/wc_i/s", "E/wc_e/s"],
        IPS=1, ISP=2, ILP=0, ISW=2, IPLT=9, RL0=-2.0, RL1=4.0, bidirectional=True)
ode.run(starting_point="LP1", origin="s_e:ss:2", name="s_e/s_i:lp2", ICP=["I/wc_i/s", "E/wc_e/s"],
        IPS=1, ISP=2, ILP=0, ISW=2, IPLT=9, RL0=-2.0, RL1=4.0, bidirectional=True)

# post-processing and plotting
##############################

# save results as csv
# ode.get_summary("s_e:ss").to_csv("steady_state_3.csv")
# ode.get_summary("s_e:ss:2").to_csv("steady_state_4.csv")
# ode.get_summary("s_e:lc").to_csv("limit_cycle_3.csv")
# ode.get_summary("s_e:lc:2").to_csv("limit_cycle_4.csv")

ode.close_session(clear_files=True)

# plot the 1D bifurcation diagram for S_I > 0
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
plt.suptitle("S_i = 1.5, a = 36, e = 55")
plt.tight_layout()
fig.canvas.draw()
plt.savefig("wc_codim1_3.pdf")

# plot the 1D bifurcation diagram for S_I = 0
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
plt.suptitle("S_i = 0, a = 36, e = 55")
plt.tight_layout()
fig.canvas.draw()
plt.savefig("wc_codim1_4.pdf")

# plot the 2D bifurcation diagram
fig, ax = plt.subplots(figsize=(12, 6))
ode.plot_continuation("E/wc_e/s", "I/wc_i/s", cont="s_e/s_i:hb1", ax=ax, color="red", ignore=["UZ"],
                      line_style_unstable="solid")
ode.plot_continuation("E/wc_e/s", "I/wc_i/s", cont="s_e/s_i:hb2", ax=ax, color="red", ignore=["UZ"],
                      line_style_unstable="solid")
ode.plot_continuation("E/wc_e/s", "I/wc_i/s", cont="s_e/s_i:lp1", ax=ax, color="blue", ignore=["UZ"],
                      line_style_unstable="solid")
ode.plot_continuation("E/wc_e/s", "I/wc_i/s", cont="s_e/s_i:lp2", ax=ax, color="blue", ignore=["UZ"],
                      line_style_unstable="solid")
ax.set_title("a = 16, e = 15")
ax.set_xlabel("S_e")
ax.set_ylabel("S_i")
plt.tight_layout()
fig.canvas.draw()
plt.savefig("wc_codim2_2.pdf")

plt.show()
