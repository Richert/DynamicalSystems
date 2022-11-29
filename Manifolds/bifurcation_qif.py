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
n_dim = 3
n_params = 7
a = ODESystem("config", auto_dir=auto_dir, init_cont=False)

# initial continuation in time to converge to fixed point
a.run(e='qif', c='ivp', name='t', DS=1e-4, DSMIN=1e-10, EPSL=1e-06, NPR=1000, NPAR=n_params, NDIM=n_dim,
      EPSU=1e-06, EPSS=1e-05, DSMAX=0.1, NMX=50000, UZR={14: 500.0}, STOP={'UZ1'})

########################
# bifurcation analysis #
########################

# 1D continuations
##################

NPR = 20

# continuation in background input
a.run(starting_point='UZ1', c='qif', ICP=1, NPAR=n_params, NDIM=n_dim, name='eta:1',
      origin='t', NMX=8000, DSMAX=0.1, UZR={1: [-2.0]}, STOP=[], NPR=NPR, RL1=20.0, RL0=-20.0,
      bidirectional=True)

# continuation in SFA strength
a.run(starting_point='UZ1', c='qif', ICP=3, NPAR=n_params, NDIM=n_dim, name='alpha:1',
      origin='eta:1', NMX=8000, DSMAX=0.01, UZR={3: [0.2]}, STOP=[], NPR=NPR, RL1=1.0, RL0=0.0,
      bidirectional=True)

# continuation of delta
a.run(starting_point='UZ1', c='qif', ICP=5, NPAR=n_params, NDIM=n_dim, name='delta:1',
      origin='alpha:1', NMX=8000, DSMAX=0.1, UZR={5: [0.2]}, STOP=[], NPR=NPR, RL1=4.0, RL0=0.01,
      bidirectional=True)

# continuation in global coupling strength
a.run(starting_point='UZ1', c='qif', ICP=2, NPAR=n_params, NDIM=n_dim, name='J:1',
      origin='delta:1', NMX=8000, DSMAX=0.1, UZR={2: [8.0]}, STOP=[], NPR=NPR, RL1=50.0, RL0=0.0,
      bidirectional=True)

# plotting
##########

import matplotlib.pyplot as plt
a.plot_continuation("PAR(2)", "U(1)", cont="J:1")
plt.show()
