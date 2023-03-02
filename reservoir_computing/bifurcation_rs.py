from pycobi import ODESystem
import matplotlib.pyplot as plt
import sys
sys.path.append('../')

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
a.run(e='rs', c='ivp', name='t', DS=1e-4, DSMIN=1e-10, EPSL=1e-06, NPR=1000, NPAR=n_params, NDIM=n_dim,
      EPSU=1e-06, EPSS=1e-05, DSMAX=0.1, NMX=50000, UZR={14: 500.0}, STOP={'UZ1'})

########################
# bifurcation analysis #
########################

# 1D continuations
##################

NPR = 20

# continuation in global coupling strength
a.run(starting_point='UZ1', c='qif', ICP=4, NPAR=n_params, NDIM=n_dim, name='g:1',
      origin='t', NMX=8000, DSMAX=0.1, UZR={4: [15.0]}, STOP=[], NPR=NPR, RL1=50.0, RL0=0.0)

# continuation in Delta
a.run(starting_point='UZ1', c='qif', ICP=5, NPAR=n_params, NDIM=n_dim, name='Delta:1',
      origin='g:1', NMX=8000, DSMAX=0.05, UZR={5: [0.5]}, STOP=[], NPR=NPR, RL1=2.0, RL0=0.0,
      bidirectional=True)

# continuation in SFA strength
a.run(starting_point='UZ1', c='qif', ICP=16, NPAR=n_params, NDIM=n_dim, name='d:1',
      origin='Delta:1', NMX=8000, DSMAX=0.1, UZR={16: [100.0]}, STOP=[], NPR=NPR, RL1=210.0, RL0=0.0,
      bidirectional=True)

# continuation in background input
a.run(starting_point='UZ1', c='qif', ICP=8, NPAR=n_params, NDIM=n_dim, name='eta:1',
      origin='Delta:1', NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=NPR, RL1=200.0, RL0=-200.0)
a.run(starting_point='UZ1', c='qif', ICP=8, NPAR=n_params, NDIM=n_dim, name='eta:2',
      origin='d:1', NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=NPR, RL1=200.0, RL0=-200.0)

# 2D continuations
##################

NPR = 20

# 2D continuation follow-up in Delta and eta for d = 10
a.run(starting_point='LP1', c='qif2', ICP=[5, 8], name='Delta/eta:lp1', origin=f'eta:1', NMX=8000, DSMAX=0.05,
      NPR=NPR, RL1=100.0, RL0=0.0, bidirectional=True)
a.run(starting_point='LP2', c='qif2', ICP=[5, 8], name='Delta/eta:lp2', origin=f'eta:1', NMX=8000, DSMAX=0.05,
      NPR=NPR, RL1=100.0, RL0=0.0, bidirectional=True)

# 2D continuation follow-up in Delta and eta for d = 100
a.run(starting_point='LP1', c='qif2', ICP=[5, 8], name='Delta/eta:lp3', origin=f'eta:2', NMX=8000, DSMAX=0.05,
      NPR=NPR, RL1=100.0, RL0=0.0, bidirectional=True)
a.run(starting_point='LP2', c='qif2', ICP=[5, 8], name='Delta/eta:lp4', origin=f'eta:2', NMX=8000, DSMAX=0.05,
      NPR=NPR, RL1=100.0, RL0=0.0, bidirectional=True)
a.run(starting_point='HB1', c='qif2', ICP=[5, 8], name='Delta/eta:hb1', origin=f'eta:2', NMX=8000, DSMAX=0.05,
      NPR=NPR, RL1=100.0, RL0=0.0, bidirectional=True)

# save results
a.to_file("../results/rs_bifurcations.pkl")
