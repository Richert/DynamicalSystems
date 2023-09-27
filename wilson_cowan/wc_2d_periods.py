import numpy as np
import matplotlib.pyplot as plt
from pycobi import ODESystem
from pyrates import CircuitTemplate
import pandas as pd

# initialize PyCoBi with steady-state solution
##############################################

# initialize pyrates model
net = CircuitTemplate.from_yaml("wc/wc")

# define extrinsic input for numerical simulation
dt = 1e-4
T = 100.0
steps = int(T/dt)
inp = np.zeros((steps,))

net.update_var(node_vars={"E/wc_e/s": 0.0, "I/wc_i/s": 1.0})

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
                    "EPSS": 1e-9, "DSMIN": 1e-12, "DSMAX": 2e-2, "IADS": 1, "THL": {}, "THU": {}}

# get limit cycle periods in S_e/S_i space
##########################################

n = 60
s_i_vals = np.round(np.linspace(-2.0, 4.0, num=n), decimals=2)
s_e_vals = np.round(np.linspace(1.0, 5.5, num=n), decimals=2)
results = {"period": [], "S_e": [], "S_i": []}

# change value of S_i for isolated WC oscillator
ode.run(c="ivp", name="s_e:ss", ICP="E/wc_e/s", IPS=1, ILP=0, ISP=2, ISW=1, RL0=0.0,
        RL1=6.0, UZR={"E/wc_e/s": s_e_vals}, DS=1e-3, **algorithm_params)

# perform a 1D parameter continuation in S_e for each user point in S_i
for i in range(n):
    res, s = ode.run(starting_point=f"UZ{i+1}", origin="s_e:ss", bidirectional=True, name=f"s_i:ss:{i+1}", STOP={"BP1"},
                     ICP="I/wc_i/s", IPS=1, ILP=0, ISP=2, ISW=1, RL0=-2.0, RL1=4.0, NPR=50, DS=1e-3, UZR={})
    if "HB" in res["bifurcation"].values:
        res2, _ = ode.run(starting_point="HB1", origin=s, name=f"s_i:lc:{i+1}", ICP="I/wc_i/s", ISW=-1, IPS=2, ISP=2,
                          STOP={"BP1", "LP3"}, NPR=10, NMX=2000, get_period=True, variables=[],
                          params=["E/wc_e/s", "I/wc_i/s"], UZR={"I/wc_i/s": s_i_vals})
        for point in res2.index:
            if res2.at[point, "bifurcation"] == "UZ" and res2.at[point, "stability"]:
                results["period"].append(res2.at[point, "period"])
                s_e = res2.at[point, "E/wc_e/s"]
                s_i = res2.at[point, "I/wc_i/s"]
                results["S_e"].append(s_e)
                results["S_i"].append(s_i)

# save results
##############

df = pd.DataFrame.from_dict(results)
df.to_csv("wc_2d_periods.csv")
ode.close_session(clear_files=True)
