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
m = 10
n_dim = 3*m+1
n_params = 20
a = PyAuto("config", auto_dir=auto_dir)

# initial continuation in time to converge to fixed point
t_sols, t_cont = a.run(e='rs_mpa', c='ivp', name='t', DS=1e-4, DSMIN=1e-10, EPSL=1e-06, NPR=1000, NPAR=n_params, NDIM=n_dim,
                       EPSU=1e-06, EPSS=1e-05, DSMAX=0.1, NMX=50000, UZR={14: 500.0}, STOP={'UZ1'})

########################
# bifurcation analysis #
########################

# main continuation
###################

# continuation in synaptic strength
c1_sols, c1_cont = a.run(starting_point='UZ1', c='mpa', ICP=4, NPAR=n_params, NDIM=n_dim, name='g:1',
                         origin=t_cont, NMX=8000, DSMAX=0.1, UZR={4: [15.0]}, STOP=[f'UZ1'], NPR=100, RL1=100.0)

# continuation in adaptation strength
vals = [25.0, 100.0]
c2_sols, c2_cont = a.run(starting_point='UZ1', c='mpa', ICP=19, NPAR=n_params, NDIM=n_dim, name='d:1',
                         origin=c1_cont, NMX=8000, DSMAX=0.1, UZR={19: vals}, STOP=[f'UZ{len(vals)}'], NPR=100,
                         RL1=200.0)

# continuation in input
for i, v in enumerate(vals):
    _, c = a.run(starting_point=f'UZ{i+1}', c='mpa', ICP=16, NPAR=n_params, NDIM=n_dim, name=f'I:{i+1}', origin=c2_cont,
                 NMX=8000, DSMAX=0.1, NPR=10, RL1=100.0)

# 2D continuation follow-up I
a.run(starting_point='LP1', c='qif2', ICP=[19, 16], name='d/I:lp1', origin=f'I:1', NMX=8000, DSMAX=0.1,
      NPR=20, RL1=200.0, RL0=0.0, bidirectional=True, NDIM=n_dim, NPAR=n_params, EPSS=1e-10)
a.run(starting_point='LP2', c='qif2', ICP=[19, 16], name='d/I:lp2', origin=f'I:1', NMX=8000, DSMAX=0.1,
      NPR=20, RL1=200.0, RL0=0.0, bidirectional=True, NDIM=n_dim, NPAR=n_params, EPSS=1e-10)
a.run(starting_point='HB1', c='qif2', ICP=[19, 16], name='d/I:hb1', origin=f'I:2', NMX=8000, DSMAX=0.1,
      NPR=20, RL1=200.0, RL0=0.0, bidirectional=True, NDIM=n_dim, NPAR=n_params, EPSS=1e-10)
a.run(starting_point='HB2', c='qif2', ICP=[19, 16], name='d/I:hb2', origin=f'I:2', NMX=8000, DSMAX=0.1,
      NPR=20, RL1=200.0, RL0=0.0, bidirectional=True, NDIM=n_dim, NPAR=n_params, EPSS=1e-10)

# save results
fname = '../results/sfa_mpa.pkl'
kwargs = {'d': vals}
a.to_file(fname, **kwargs)
