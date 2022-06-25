from pyauto import PyAuto
import sys

"""
Bifurcation analysis of the Izhikevich mean-field model.

To run this code, you need Python >= 3.6 with PyRates (https://github.com/pyrates-neuroscience/PyRates) and 
auto-07p (https://github.com/auto-07p/auto-07p) installed.
"""

# preparations
##############

path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"

# config
n_dim = 4
n_params = 17
a = PyAuto("config", auto_dir=auto_dir)

# initial continuation in time to converge to fixed point
t_sols, t_cont = a.run(e='ik', c='ivp', name='t', DS=1e-4, DSMIN=1e-10, EPSL=1e-06, NPR=1000, NPAR=n_params, NDIM=n_dim,
                       EPSU=1e-06, EPSS=1e-05, DSMAX=0.1, NMX=50000, UZR={14: 500.0}, STOP={'UZ1'})

########################
# bifurcation analysis #
########################

# main continuation
###################

# continuation in background input
c1_sols, c1_cont = a.run(starting_point='UZ1', c='qif', ICP=5, NPAR=n_params, NDIM=n_dim, name='eta:1',
                         origin=t_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=20, RL1=200.0, RL0=-200.0,
                         bidirectional=True)

# continuation of limit cycle
a.run(starting_point='HB1', c='qif2b', ICP=[5, 11], NPAR=n_params, NDIM=n_dim, name='eta:1:lc', origin=c1_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=20, RL1=200.0, RL0=-200.0)

# 2D continuation follow-up
a.run(starting_point='LP1', c='qif2', ICP=[4, 5], name='g/eta:lp1', origin=f'eta:1', NMX=8000, DSMAX=0.05,
      NPR=10, RL1=100.0, RL0=0.0, bidirectional=True)
a.run(starting_point='LP2', c='qif2', ICP=[4, 5], name='g/eta:lp2', origin=f'eta:1', NMX=8000, DSMAX=0.05,
      NPR=10, RL1=100.0, RL0=0.0, bidirectional=True)
a.run(starting_point='HB1', c='qif2', ICP=[4, 5], name='g/eta:hb1', origin=f'eta:1', NMX=8000, DSMAX=0.05,
      NPR=10, RL1=100.0, RL0=0.0, bidirectional=True)

# save results
fname = '../results/ik_bifs.pkl'
kwargs = {}
a.to_file(fname, **kwargs)
