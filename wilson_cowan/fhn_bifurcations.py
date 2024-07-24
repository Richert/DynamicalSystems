import numpy as np
import matplotlib.pyplot as plt
from pycobi import ODESystem
from pyrates import CircuitTemplate


# get a steady-state solution to start from
###########################################

# # initialize pyrates model
# net = CircuitTemplate.from_yaml("fhn/fhn")
#
# # define extrinsic input for numerical simulation
# dt = 1e-3
# T = 1000.0
# steps = int(T/dt)
# inp = np.zeros((steps,))
#
# # perform numerical simulation to receive well-defined initial condition
# res = net.run(simulation_time=T, step_size=dt, inputs={"p/fhn_op/I_ext": inp},
#               outputs={"v": "p/fhn_op/v", "u": "p/fhn_op/u"}, solver="scipy", method="RK45",
#               in_place=False, vectorize=False, rtol=1e-8, atol=1e-8)
#
# # plot results
# res.plot()
# plt.show()
#
# # generate pycobi files with initial condition
# _, _, params, state_vars = net.get_run_func(func_name="fhn_rhs", file_name="fhn", step_size=dt, auto=True,
#                                             backend="fortran", solver='scipy', vectorize=False,
#                                             float_precision='float64', in_place=False)

# initialize pycobi instance
ode = ODESystem(eq_file="fhn", auto_dir="~/PycharmProjects/auto-07p", init_cont=False)

# define the standard auto parameters
algorithm_params = {"NTST": 200, "NCOL": 4, "IAD": 3, "IPLT": 0, "NBC": 0, "NINT": 0, "NMX": 8000, "NPR": 20,
                    "MXBF": 7, "IID": 2, "ITMX": 40, "ITNW": 40, "NWTN": 5, "JAC": 0, "EPSL": 1e-6, "EPSU": 1e-6,
                    "EPSS": 1e-4, "DS": 1e-3, "DSMIN": 1e-10, "DSMAX": 5e-2, "IADS": 1, "THL": {11: 0.0}, "THU": {}}

# perform a 1D parameter continuation in a
a_col = [0.4, 0.6, 0.8]
ode.run(bidirectional=True, name="a:ss", c="fhn", ICP=[5], RL0=0.0, RL1=1.0, NDIM=2, NPAR=5,
        UZR={5: a_col}, **algorithm_params)

I_min, I_max = -10.0, 10.0
for i in range(len(a_col)):

    # perform a 1D parameter continuation in I_ext
    ss_key = f"I_ext:ss:{i+1}"
    ode.run(starting_point=f"UZ{i+1}",origin="a:ss", bidirectional=True, name=ss_key, ICP=[1],
            RL0=-10.0, RL1=10.0)

    # continue limit cycle solution
    ode.run(starting_point="HB1", origin=ss_key, name=f"{ss_key}:lc1", ISW=-1, IPS=2, ISP=2, STOP={"BP2", "LP2"},
            NMX=4000, get_period=True)
    ode.run(starting_point="HB2", origin=ss_key, name=f"{ss_key}:lc2", ISW=-1, IPS=2, ISP=2, STOP={"BP2", "LP2"},
            NMX=4000, get_period=True)

    # perform a 2D parameter continuation of the hopf bifurcation
    ode.run(starting_point="HB1",origin=ss_key, bidirectional=True, name=f"HB1:b/I_ext:{i+1}", ICP=[4, 1], RL0=0.0,
            RL1=5.0, IPS=1, ISP=2, ILP=0, ISW=2, UZR={4: [2.0]})
    ode.run(starting_point="HB2", origin=ss_key, bidirectional=True, name=f"HB2:b/I_ext:{i + 1}", ICP=[4, 1], RL0=0.0,
            RL1=5.0, IPS=1, ISP=2, ILP=0, ISW=2)

    # perform a second 1D parameter continuation in I_ext
    ode.run(starting_point=f"UZ1", origin=f"HB1:b/I_ext:{i+1}", bidirectional=True, name="a:ss:2", ICP=[1],
            RL0=-10.0, RL1=10.0, UZR={}, ILP=1, IPS=1, ISP=1, ISW=1)

    # perform a 2D parameter continuation of the fold bifurcation
    ode.run(starting_point="LP1", origin="a:ss:2", bidirectional=True, name=f"LP1:b/I_ext:{i+1}", ICP=[4, 1], RL0=0.0,
            RL1=3.0, IPS=1, ISP=2, ILP=0, ISW=2, UZR={})

# plotting
fig, axes = plt.subplots(ncols=len(a_col), figsize=(12, 4))
fig2, axes2 = plt.subplots(ncols=len(a_col), figsize=(12, 4))
for i, ax in enumerate(axes):
    ode.plot_continuation("PAR(1)", "PAR(4)", ax=ax, cont=f"HB1:b/I_ext:{i+1}")
    ode.plot_continuation("PAR(1)", "PAR(4)", ax=ax, cont=f"HB2:b/I_ext:{i+1}")
    ode.plot_continuation("PAR(1)", "PAR(4)", ax=ax, cont=f"LP1:b/I_ext:{i+1}")
    ax.set_xlabel("I_ext")
    ax.set_ylabel("b")
    ax.set_title(f"a = {a_col[i]}")
    ax.set_xlim([I_min, I_max])
    plt.tight_layout()
    ode.plot_continuation("PAR(1)", "U(1)", ax=axes2[i], cont=f"I_ext:ss:{i+1}")
    ode.plot_continuation("PAR(1)", "U(1)", ax=axes2[i], cont=f"I_ext:ss:{i + 1}:lc1")
    ode.plot_continuation("PAR(1)", "U(1)", ax=axes2[i], cont=f"I_ext:ss:{i + 1}:lc2")
    axes2[i].set_xlabel("I_ext")
    axes2[i].set_ylabel("v")
    plt.tight_layout()
plt.show()
