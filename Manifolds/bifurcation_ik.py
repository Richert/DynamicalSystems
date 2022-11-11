from pycobi import ODESystem
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
a = ODESystem("config", auto_dir=auto_dir, init_cont=False)

# initial continuation in time to converge to fixed point
a.run(e='ik', c='ivp', name='t', DS=1e-4, DSMIN=1e-10, EPSL=1e-06, NPR=1000, NPAR=n_params, NDIM=n_dim,
      EPSU=1e-06, EPSS=1e-05, DSMAX=0.1, NMX=50000, UZR={14: 500.0}, STOP={'UZ1'})

########################
# bifurcation analysis #
########################

# 1D continuations
##################

NPR = 20

# continuation in background input
a.run(starting_point='UZ1', c='qif', ICP=5, NPAR=n_params, NDIM=n_dim, name='eta:1',
      origin='t', NMX=8000, DSMAX=0.1, UZR={5: [50.0]}, STOP=[], NPR=NPR, RL1=200.0, RL0=-200.0,
      bidirectional=True)

# continuation in SFA strength
a.run(starting_point='UZ1', c='qif', ICP=16, NPAR=n_params, NDIM=n_dim, name='d:1',
      origin='eta:1', NMX=8000, DSMAX=0.1, UZR={16: [150.0]}, STOP=[], NPR=NPR, RL1=200.0, RL0=0.0,
      bidirectional=True)

# continuation in global coupling strength
a.run(starting_point='UZ1', c='qif', ICP=4, NPAR=n_params, NDIM=n_dim, name='g:2',
      origin='d:1', NMX=8000, DSMAX=0.1, UZR={4: [20.0]}, STOP=[], NPR=NPR, RL1=50.0, RL0=0.0,
      bidirectional=True)

# continuation of delta
a.run(starting_point='UZ1', c='qif', ICP=6, NPAR=n_params, NDIM=n_dim, name='delta:1',
      origin='g:2', NMX=8000, DSMAX=0.1, UZR={6: [1.0]}, STOP=[], NPR=NPR, RL1=5.0, RL0=0.01,
      bidirectional=True)

# continuation of b
a.run(starting_point='UZ1', c='qif', ICP=9, NPAR=n_params, NDIM=n_dim, name='b:1',
      origin='delta:1', NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=NPR, RL1=10.0, RL0=-10.0,
      bidirectional=True)

# 2D continuations
##################

NPR = 20

# 2D continuation follow-up in d and eta
a.run(starting_point='LP1', c='qif2', ICP=[16, 5], name='d/eta:lp1', origin=f'd:1', NMX=8000, DSMAX=0.05,
      NPR=NPR, RL1=200.0, RL0=0.0, bidirectional=True)
# a.run(starting_point='LP2', c='qif2', ICP=[16, 5], name='d/eta:lp2', origin=f'd:1', NMX=8000, DSMAX=0.05,
#       NPR=NPR, RL1=200.0, RL0=0.0, bidirectional=True)
a.run(starting_point='HB1', c='qif2', ICP=[16, 5], name='d/eta:hb1', origin=f'd:1', NMX=8000, DSMAX=0.05,
      NPR=NPR, RL1=200.0, RL0=0.0, bidirectional=True)

# 2D continuation follow-up in g and eta
a.run(starting_point='LP1', c='qif2', ICP=[4, 5], name='g/eta:lp1', origin=f'g:2', NMX=8000, DSMAX=0.05,
      NPR=NPR, RL1=100.0, RL0=0.0, bidirectional=True)
a.run(starting_point='LP2', c='qif2', ICP=[4, 5], name='g/eta:lp2', origin=f'g:2', NMX=8000, DSMAX=0.05,
      NPR=NPR, RL1=100.0, RL0=0.0, bidirectional=True)
a.run(starting_point='HB1', c='qif2', ICP=[4, 5], name='g/eta:hb1', origin=f'g:2', NMX=8000, DSMAX=0.05,
      NPR=NPR, RL1=100.0, RL0=0.0, bidirectional=True)

# save results
fname = '../results/ik_bifs.pkl'
kwargs = {}
a.to_file(fname, **kwargs)
