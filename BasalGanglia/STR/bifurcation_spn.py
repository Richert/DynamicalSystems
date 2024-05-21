from pycobi import ODESystem
import sys

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
                         origin=t_cont, NMX=8000, DSMAX=0.05, UZR={4: [5.0]}, STOP=[f'UZ1'], NPR=100,
                         RL1=1000.0, RL0=0.0)

# continuation in Delta
c2_sols, c2_cont = a.run(starting_point='UZ1', c='qif', ICP=7, NPAR=n_params, NDIM=n_dim, name='D:1',
                         origin=c1_cont, NMX=8000, DSMAX=0.05, UZR={7: [1.0]}, STOP=[f'UZ1'], NPR=100,
                         RL1=3.0, RL0=0.0, bidirectional=True)

# main continuations
####################

# continuation in I
r1_sols, r1_cont = a.run(starting_point='UZ1', c='qif', ICP=16, NPAR=n_params, NDIM=n_dim, name='I_ext:1',
                         origin=c2_cont, NMX=8000, DSMAX=0.5, UZR={}, STOP=[], NPR=100,
                         RL1=500.0, RL0=0.0, bidirectional=True)
a.run(starting_point='HB1', c='qif2', ICP=[9, 16], name='E/I:lp3', origin=r1_cont, NMX=8000, DSMAX=0.05,
      NPR=20, RL1=-50.0, RL0=-70.0, bidirectional=True)
