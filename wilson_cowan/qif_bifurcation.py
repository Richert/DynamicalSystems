import numpy as np
import matplotlib.pyplot as plt
from pycobi import ODESystem

# initialize pycobi instance
ode = ODESystem(eq_file="qif_sd", auto_dir="~/PycharmProjects/auto-07p", init_cont=False)

# define the standard auto parameters
algorithm_params = {"NTST": 200, "NCOL": 4, "IAD": 3, "IPLT": 0, "NBC": 0, "NINT": 0, "NMX": 8000, "NPR": 20,
                    "MXBF": 7, "IID": 2, "ITMX": 40, "ITNW": 40, "NWTN": 5, "JAC": 0, "EPSL": 1e-8, "EPSU": 1e-8,
                    "EPSS": 1e-7, "DS": 1e-3, "DSMIN": 1e-10, "DSMAX": 5e-2, "IADS": 1, "THL": {}, "THU": {}}

# increase alpha
ode.run(name="alpha:ss", c="fhn", ICP=[3], RL0=0.0, RL1=1.0, NDIM=4, NPAR=5,
        UZR={3: [0.08]}, STOP=["UZ1"], **algorithm_params)

# perform a 1D parameter continuation in J
Js = [16.0, 17.0, 18.0, 19.0, 20.0]
ode.run(bidirectional=True, starting_point="UZ1", origin="alpha:ss", ICP=[2], RL0=0.0, RL1=30.0, UZR={2: Js},
        name="J:ss", STOP=[])

periods, etas = [], []
for i in range(len(Js)):

    # continuation in eta
    _, c = ode.run(bidirectional=True, starting_point=f"UZ{i+1}", origin="J:ss", ICP=[1], RL0=-10.0, RL1=20.0, UZR={},
                   name=f"eta:ss:{i+1}")

    # continuation of limit cycles
    s1, _ = ode.run(starting_point="HB1", origin=c, name=f"eta:ss:{i+1}:lc1", ISW=-1, IPS=2, ISP=2, STOP={"BP1", "LP2"},
                    NMX=4000, get_period=True)
    # s2, _ = ode.run(starting_point="HB2", origin=c, name=f"eta:ss:{i + 1}:lc2", ISW=-1, IPS=2, ISP=2, STOP={"BP1", "LP2"},
    #                 NMX=4000, get_period=True)

    # extract periods
    periods.append(s1["period"].values.tolist()) #+ s2["period"].values.tolist())
    etas.append(s1["PAR(1)"].values.tolist()) #+ s2["PAR(1)"].values.tolist())

# plotting
fig, ax = plt.subplots(figsize=(8, 4))
for J, p, eta in zip(Js, periods, etas):
    ax.plot(eta, p, label=f"J = {J}")
ax.set_xlabel("eta")
ax.set_ylabel("period")
ax.legend()
plt.tight_layout()
plt.show()
