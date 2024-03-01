from pycobi import ODESystem
import sys
import matplotlib.pyplot as plt

"""
Bifurcation analysis of a QIF population with distributed etas and exponential synpatic dynamics.
"""

# preparations
##############

path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"

# config
n_dim = 3
n_params = 4
a = ODESystem("qif_inh", working_dir="auto_files", auto_dir=auto_dir, init_cont=False)

# initial continuation in time to converge to fixed point
t_sols, t_cont = a.run(c='ivp', name='t', DS=1e-4, DSMIN=1e-10, EPSL=1e-06, NPR=1000, NPAR=n_params, NDIM=n_dim,
                       EPSU=1e-06, EPSS=1e-05, DSMAX=0.1, NMX=50000, UZR={14: 2000.0}, STOP={'UZ1'})

# bifurcation analysis
######################

# continuation in synaptic strength
c1_sols, c1_cont = a.run(starting_point='UZ1', c='qif', ICP=2, name='J', origin=t_cont,
                         UZR={2: [-20.0]}, STOP=[f'UZ1'], RL0=-50.0, RL1=0.0, NPAR=n_params, NDIM=n_dim, DS="-")

# continuation in input strength
c2_sols, c2_cont = a.run(starting_point='UZ1', ICP=1, name='eta', origin=c1_cont, UZR={}, STOP=[], RL1=200.0, RL0=0.0,
                         bidirectional=True)

# continuation of limit cycle
a.run(starting_point='HB1', c='qif2b', ICP=1, name='eta:lc', origin=c2_cont, UZR={}, STOP=[], RL1=200.0, RL0=0.0)

# continuation of Hopf curve
a.run(starting_point='HB1', c='qif2', ICP=[3, 1], name='tau/eta', origin=c2_cont, RL1=50.0, RL0=0.0, bidirectional=True,
      NPAR=n_params, NDIM=n_dim, STOP=[])

# plotting
##########

fig, axes = plt.subplots(figsize=(12, 6), nrows=2)

# plot main continuation
ax = axes[0]
a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta', ax=ax, bifurcation_legend=False)
a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta:lc', ax=ax, bifurcation_legend=True, line_color_stable="orange")
ax.set_xlabel("input")
ax.set_ylalbel("r")
ax.set_title("1D bifurcation diagram")

# plot 2D continuation
ax = axes[1]
a.plot_continuation('PAR(1)', 'PAR(3)', cont='tau/eta', ax=ax, bifurcation_legend=True)
ax.set_xlabel("input")
ax.set_ylabel("tau")
ax.set_title("2D bifurcation diagram")

plt.tight_layout()
plt.show()
