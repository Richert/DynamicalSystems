import numpy as np
import matplotlib.pyplot as plt
from pycobi import ODESystem
from pyrates import CircuitTemplate


# get a steady-state solution to start from
###########################################

# initialize pyrates model
net = CircuitTemplate.from_yaml("fhn/fhn")

# define extrinsic input for numerical simulation
dt = 1e-3
T = 1000.0
steps = int(T/dt)
inp = np.zeros((steps,))

# perform numerical simulation to receive well-defined initial condition
res = net.run(simulation_time=T, step_size=dt, inputs={"p/fhn_op/I_ext": inp},
              outputs={"v": "p/fhn_op/v", "u": "p/fhn_op/u"}, solver="scipy", method="RK45",
              in_place=False, vectorize=False, rtol=1e-8, atol=1e-8)

# plot results
res.plot()
plt.show()

# generate pycobi files with initial condition
_, _, params, state_vars = net.get_run_func(func_name="fhn_rhs", file_name="fhn", step_size=dt, auto=True,
                                            backend="fortran", solver='scipy', vectorize=False,
                                            float_precision='float64', in_place=False)

# initialize pycobi instance
ode = ODESystem(eq_file="fhn", auto_dir="~/PycharmProjects/auto-07p", init_cont=False, params=params[3:],
                state_vars=list(state_vars))

# define the standard auto parameters
algorithm_params = {"NTST": 400, "NCOL": 4, "IAD": 3, "IPLT": 0, "NBC": 0, "NINT": 0, "NMX": 8000, "NPR": 5,
                    "MXBF": 7, "IID": 2, "ITMX": 40, "ITNW": 40, "NWTN": 12, "JAC": 0, "EPSL": 1e-8, "EPSU": 1e-8,
                    "EPSS": 1e-9, "DS": 1e-3, "DSMIN": 1e-12, "DSMAX": 1e-2, "IADS": 1, "THL": {}, "THU": {}}

# perform bifurcation analysis for S_i = 1.35
#############################################

# perform a 1D parameter continuation in S_e
ode.run(bidirectional=True, name="I_ext:ss", c="ivp", ICP="p/fhn_op/I_ext",
        IPS=1, ILP=1, ISP=2, ISW=1, RL0=-5.0, RL1=5.0, **algorithm_params, UZR={})
