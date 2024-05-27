from pycobi import ODESystem
import sys
import matplotlib.pyplot as plt

"""
Bifurcation analysis of the Izhikevich mean-field model for a single SPN population.

To run this code, you need Python >= 3.6 with PyRates (https://github.com/pyrates-neuroscience/PyRates) and 
auto-07p (https://github.com/auto-07p/auto-07p) installed.
"""

# preparations
##############

path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"

# config
n_dim = 5
n_params = 25
a = ODESystem.from_yaml("config/mf/spn", auto_dir=auto_dir, working_dir="config", init_cont=False)

# initial continuation in time to converge to fixed point
t_sols, t_cont = a.run(c='ivp', name='t', DS=1e-4, DSMIN=1e-10, EPSL=1e-06, NPR=1000, NPAR=n_params, NDIM=n_dim,
                       EPSU=1e-06, EPSS=1e-05, DSMAX=0.5, NMX=50000, UZR={14: 3000.0}, STOP={'UZ1'})

########################
# bifurcation analysis #
########################

# prepare state
###############

# continuation in Delta
c1_sols, c1_cont = a.run(starting_point='UZ1', c='qif', ICP=7, NPAR=n_params, NDIM=n_dim, name='D:0',
                         origin=t_cont, NMX=8000, DSMAX=0.05, UZR={7: [0.2]}, STOP=[f'UZ1'], NPR=100,
                         RL1=3.0, RL0=0.0, bidirectional=True)

# continuation in synaptic strength
gs = [2.0, 4.0, 8.0, 16.0]
c2_sols, c2_cont = a.run(starting_point='UZ1', c='qif', ICP=4, NPAR=n_params, NDIM=n_dim, name='g:0',
                         origin=c1_cont, NMX=8000, DSMAX=0.01, UZR={4: gs}, STOP=[f'UZ4'], NPR=10,
                         RL1=20.0, RL0=0.0, bidirectional=True)

# main continuations
####################

for i in range(len(gs)):

    # 1D continuation in I
    r1_sols, r1_cont = a.run(starting_point=f'UZ{i+1}', c='qif', ICP=17, NPAR=n_params, NDIM=n_dim, name=f'I_ext:{i+1}',
                             origin=c2_cont, NMX=8000, DSMAX=0.5, UZR={"17": [400.0]}, STOP=[], NPR=100,
                             RL1=1000.0, RL0=0.0, bidirectional=True)

    # 2D continuation in D and I
    a.run(starting_point='HB1', c='qif2', ICP=[7, 17], name=f'D/I:{i+1}', origin=r1_cont, NMX=8000, DSMAX=0.1,
          NPR=50, RL1=1.0, RL0=0.0, bidirectional=True, NDIM=n_dim, NPAR=n_params)

# 1D continuation in D
r2_sols, r2_cont = a.run(starting_point=f'UZ1', c='qif', ICP=7, NPAR=n_params, NDIM=n_dim, name=f'D:1',
                         origin="I_ext:2", NMX=8000, DSMAX=0.01, UZR={}, STOP=[], NPR=20, RL1=0.5, RL0=0.05,
                         bidirectional=True)

# limit cycle continuation in D
a.run(starting_point='HB1', ICP=7, name=f'D:lc:1', origin=r2_cont, NMX=8000, DSMAX=0.01,
      NPR=20, RL1=0.5, RL0=0.05, NDIM=n_dim, NPAR=n_params, ISW=-1, ISP=2, IPS=2, ILP=1)

# plotting
##########

# D vs I
fig, ax = plt.subplots(figsize=(6, 6))
lines = []
colors = ["black", "red", "blue", "green"]
for i in range(len(gs)):
    l = a.plot_continuation("PAR(17)", "PAR(7)", cont=f"D/I:{i+1}", ax=ax, line_style_unstable="solid",
                            line_color_stable=colors[i], line_color_unstable=colors[i])
    lines.append(l)
ax.set_xlabel("I")
ax.set_ylabel("D")
ax.legend(lines, [f"g_i = {g}" for g in gs])
plt.tight_layout()

# limit cycle plot
fig, ax = plt.subplots(figsize=(10, 3))
a.plot_continuation("PAR(7)", "U(1)", cont=f"D:1", ax=ax, line_style_unstable="dotted",
                    line_color_stable="black", line_color_unstable="grey")
a.plot_continuation("PAR(7)", "U(1)", cont=f"D:lc:1", ax=ax, line_style_unstable="dotted",
                    line_color_stable="green", line_color_unstable="green", bifurcation_legend=False)
ax.set_xlim([0.08, 0.3])
ax.set_xlabel("D")
ax.set_ylabel("r")
plt.tight_layout()
plt.show()
