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
n_dim = 4
n_params = 25
a = ODESystem.from_yaml("config/mf/spn", auto_dir=auto_dir, working_dir="config", init_cont=False)

# initial continuation in time to converge to fixed point
t_sols, t_cont = a.run(c='ivp', name='t', DS=1e-4, DSMIN=1e-10, EPSL=1e-06, NPR=1000, NPAR=n_params, NDIM=n_dim,
                       EPSU=1e-06, EPSS=1e-05, DSMAX=0.1, NMX=50000, UZR={14: 3000.0}, STOP={'UZ1'})

########################
# bifurcation analysis #
########################

# prepare state
###############

# continuation in synaptic strength
c1_sols, c1_cont = a.run(starting_point='UZ1', c='qif', ICP=4, NPAR=n_params, NDIM=n_dim, name='g:1',
                         origin=t_cont, NMX=8000, DSMAX=0.01, UZR={4: [3.0, 6.0]}, STOP=[f'UZ2'], NPR=10,
                         RL1=20.0, RL0=0.0, bidirectional=True)

# continuation in Delta
c2_sols, c2_cont = a.run(starting_point='UZ1', c='qif', ICP=7, NPAR=n_params, NDIM=n_dim, name='D:1',
                         origin=c1_cont, NMX=8000, DSMAX=0.05, UZR={7: [0.2]}, STOP=[f'UZ1'], NPR=100,
                         RL1=3.0, RL0=0.0, bidirectional=True)
c3_sols, c3_cont = a.run(starting_point='UZ2', c='qif', ICP=7, NPAR=n_params, NDIM=n_dim, name='D:2',
                         origin=c1_cont, NMX=8000, DSMAX=0.05, UZR={7: [0.2]}, STOP=[f'UZ1'], NPR=100,
                         RL1=3.0, RL0=0.0, bidirectional=True)

# main continuations
####################

# continuation in I for g = 4.0
r1_sols, r1_cont = a.run(starting_point='UZ1', c='qif', ICP=16, NPAR=n_params, NDIM=n_dim, name='I_ext:1',
                         origin=c2_cont, NMX=8000, DSMAX=0.5, UZR={}, STOP=[], NPR=100,
                         RL1=1000.0, RL0=0.0, bidirectional=True)
a.run(starting_point='HB1', c='qif2', ICP=[9, 16], name='E/I:hb1', origin=r1_cont, NMX=8000, DSMAX=0.1,
      NPR=50, RL1=-50.0, RL0=-70.0, bidirectional=True)
a.run(starting_point='HB1', c='qif2', ICP=[7, 16], name='D/I:hb1', origin=r1_cont, NMX=8000, DSMAX=0.1,
      NPR=50, RL1=1.0, RL0=0.0, bidirectional=True)
a.run(starting_point='HB1', c='qif2', ICP=[4, 16], name='g/I:hb1', origin=r1_cont, NMX=8000, DSMAX=0.05,
      NPR=50, RL1=20.0, RL0=0.0, bidirectional=True)

# continuation in I for g = 6.0
r2_sols, r2_cont = a.run(starting_point='UZ1', c='qif', ICP=16, NPAR=n_params, NDIM=n_dim, name='I_ext:2',
                         origin=c3_cont, NMX=8000, DSMAX=0.5, UZR={}, STOP=[], NPR=100,
                         RL1=1000.0, RL0=0.0, bidirectional=True)
a.run(starting_point='HB1', c='qif2', ICP=[9, 16], name='E/I:hb2', origin=r2_cont, NMX=8000, DSMAX=0.1,
      NPR=50, RL1=-50.0, RL0=-70.0, bidirectional=True)
a.run(starting_point='HB1', c='qif2', ICP=[7, 16], name='D/I:hb2', origin=r2_cont, NMX=8000, DSMAX=0.1,
      NPR=50, RL1=1.0, RL0=0.0, bidirectional=True)
a.run(starting_point='HB1', c='qif2', ICP=[4, 16], name='g/I:hb2', origin=r2_cont, NMX=8000, DSMAX=0.05,
      NPR=50, RL1=20.0, RL0=0.0, bidirectional=True)

# plotting
##########

fig, axes = plt.subplots(ncols=3, figsize=(12, 4))

# E vs I
ax = axes[0]
a.plot_continuation("PAR(16)", "PAR(9)", cont="E/I:hb1", ax=ax, line_style_unstable="solid")
a.plot_continuation("PAR(16)", "PAR(9)", cont="E/I:hb2", ax=ax, line_style_unstable="solid",
                    line_color_stable="red", line_color_unstable="red")
ax.set_xlabel("I")
ax.set_ylabel("E_i")

# D vs I
ax = axes[1]
a.plot_continuation("PAR(16)", "PAR(7)", cont="D/I:hb1", ax=ax, line_style_unstable="solid")
a.plot_continuation("PAR(16)", "PAR(7)", cont="D/I:hb2", ax=ax, line_style_unstable="solid",
                    line_color_stable="red", line_color_unstable="red")
ax.set_xlabel("I")
ax.set_ylabel("D")

# g vs I
ax = axes[2]
a.plot_continuation("PAR(16)", "PAR(4)", cont="g/I:hb1", ax=ax, line_style_unstable="solid")
a.plot_continuation("PAR(16)", "PAR(4)", cont="g/I:hb2", ax=ax, line_style_unstable="solid")
ax.set_xlabel("I")
ax.set_ylabel("g_i")

plt.tight_layout()
plt.show()
