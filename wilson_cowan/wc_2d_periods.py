import numpy as np
import matplotlib.pyplot as plt
from pycobi import ODESystem
from pyrates import CircuitTemplate


# initialize PyCoBi with steady-state solution
##############################################

# initialize pyrates model
net = CircuitTemplate.from_yaml("wc/wc")

# define extrinsic input for numerical simulation
dt = 1e-4
T = 100.0
steps = int(T/dt)
inp = np.zeros((steps,))

net.update_var(node_vars={"E/wc_e/s": -2.1, "I/wc_i/s": 4.1})

# perform numerical simulation to receive well-defined initial condition
res = net.run(simulation_time=T, step_size=dt, inputs={"E/wc_e/s": inp},
              outputs={"E": "E/wc_e/u", "I": "I/wc_i/u"}, solver="scipy", method="RK23",
              in_place=False, vectorize=False, rtol=1e-8, atol=1e-8)

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
                    "EPSS": 1e-9, "DSMIN": 1e-12, "DSMAX": 5e-2, "IADS": 1, "THL": {}, "THU": {}}

# get limit cycle periods in S_e/S_i space
##########################################

n = 60
s_i_vals = np.linspace(-2.0, 4.0, num=n)

# change value of S_i for isolated WC oscillator
ode.run(c="ivp", name="s_i:ss", ICP="I/wc_i/s", IPS=1, ILP=0, ISP=2, ISW=1, RL0=-2.1,
        RL1=4.1, UZR={"I/wc_i/s": s_i_vals}, DS="-", **algorithm_params)

# perform a 1D parameter continuation in S_e for each user point in S_i
for i in range(n):
    res, s = ode.run(starting_point=f"UZ{i+1}", origin="s_i:ss", bidirectional=False, name=f"s_e:ss:{i+1}",
                     ICP="E/wc_e/s", IPS=1, ILP=0, ISP=2, ISW=1, RL1=8.0, NPR=50, DS=1e-3, UZR={})
    if "HB" in res["bifurcation"].values:
        ode.run(starting_point="HB1", origin=s, name=f"s_e:lc:{i+1}", ISW=-1, IPS=2, ISP=2, STOP={"BP1", "LP3"},
                NPR=10, DSMAX=0.1, NMX=2000, get_period=True, variables=[], params=["E/wc_e/s", "I/wc_i/s"])

# save results
##############

ode.to_file("wc_2d_periods.pkl")
ode.close_session(clear_files=True)
