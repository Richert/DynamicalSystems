from pycobi import ODESystem
import numpy as np
import matplotlib.pyplot as plt

# define the standard auto parameters
auto_params = {"NTST": 400, "NCOL": 4, "IAD": 3, "IPLT": 0, "NBC": 0, "NINT": 0, "NMX": 8000, "NPR": 100,
               "MXBF": 7, "IID": 2, "ITMX": 40, "ITNW": 40, "NWTN": 12, "JAC": 0, "EPSL": 1e-9, "EPSU": 1e-9,
               "EPSS": 1e-7, "DS": 1e-3, "DSMIN": 1e-12, "DSMAX": 5e-2, "IADS": 1, "THL": {}, "THU": {}}

# initialize ODE system
ode = ODESystem.from_yaml("ik_mf/eic", working_dir="config", auto_dir="~/PycharmProjects/auto-07p",
                          init_cont=True, NMX=50000, NPR=1000, UZR={14: 250.0}, STOP=["UZ1"])

# perform continuation in input to the RS population
ode.run(name="init1", origin=0, starting_point="UZ1", ICP="rs/rs_op/I_ext", IPS=1, ILP=1, ISP=2, ISW=1,
        RL0=-20.0, RL1=80.0, UZR={"rs/rs_op/I_ext": 60.0}, STOP=["UZ1"], **auto_params)

# perform continuation in heterogeneity of the LTS population
delta_lts = np.linspace(0.5, 2.0, num=10)
ode.run(name="init2", origin="init1", starting_point="UZ1", ICP="lts/lts_op/Delta", RL0=0.0, RL1=2.1,
        bidirectional=True, UZR={"lts/lts_op/Delta": delta_lts}, STOP=[])

for i in range(len(delta_lts)):

    # perform continuation in input to the LTS population
    ode.run(name=f"ss{i+1}", origin="init2", starting_point=f"UZ{i+1}", ICP="lts/lts_op/I_ext", RL0=-10.0, RL1=300.0,
            DS=1e-3, UZR={})

    # perform 2D continuation of detected fold bifurcation
    # ode.run(name="lp:1", origin="ss2", starting_point="LP1", ICP=["rs/rs_op/Delta", "lts/lts_op/I_ext"], IPS=1, ISP=2,
    #         ILP=0, ISW=2, bidirectional=True, RL0=0.0, RL1=3.0, DSMAX=0.01, STOP=["BP1"])
    # ode.run(name="lp:2", origin="ss2", starting_point="LP2", ICP=["rs/rs_op/Delta", "lts/lts_op/I_ext"], IPS=1, ISP=2,
    #         ILP=0, ISW=2, bidirectional=True, RL0=0.0, RL1=3.0, DSMAX=0.01, STOP=["BP1"])
    ode.run(name=f"hb{i+1}", origin=f"ss{i+1}", starting_point="HB1", ICP=["rs/rs_op/Delta", "lts/lts_op/I_ext"],
            IPS=1, ISP=2, ILP=0, ISW=2, bidirectional=True, RL0=0.0, RL1=6.0, DSMAX=0.05, STOP=["BP2"])

    # plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    # ode.plot_continuation("lts/lts_op/I_ext", "rs/rs_op/Delta", cont="lp:1", ax=ax, line_style_unstable="solid",
    #                       color="black")
    # ode.plot_continuation("lts/lts_op/I_ext", "rs/rs_op/Delta", cont="lp:2", ax=ax, line_style_unstable="solid",
    #                       color="black")
    ode.plot_continuation("lts/lts_op/I_ext", "rs/rs_op/Delta", cont=f"hb{i+1}", ax=ax,
                          line_style_unstable="solid", color="orange")
    ax.set_xlabel("I_lts")
    ax.set_ylabel("Delta_rs")
    ax.set_title(f"Delta_lts = {delta_lts[i]}")
    plt.show()
