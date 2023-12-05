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
n_params = 20
a = ODESystem("rs2", working_dir="config", auto_dir=auto_dir, init_cont=False)

# initial continuation in time to converge to fixed point
t_sols, t_cont = a.run(c='ivp', name='t', DS=1e-4, DSMIN=1e-10, EPSL=1e-06, NPR=1000, NPAR=n_params, NDIM=n_dim,
                       EPSU=1e-06, EPSS=1e-05, DSMAX=0.1, NMX=50000, UZR={14: 500.0}, STOP={'UZ1'})

########################
# bifurcation analysis #
########################

# prepare state
###############

# continuation in synaptic strength
c1_sols, c1_cont = a.run(starting_point='UZ1', c='qif', ICP=4, NPAR=n_params, NDIM=n_dim, name='g:1',
                         origin=t_cont, NMX=8000, DSMAX=0.05, UZR={4: [15.0]}, STOP=[f'UZ1'], NPR=100,
                         RL1=1000.0, RL0=0.0)

# continuation in Delta
vals1 = [0.1, 1.0]
c2_sols, c2_cont = a.run(starting_point='UZ1', c='qif', ICP=5, NPAR=n_params, NDIM=n_dim, name='D:1',
                         origin=c1_cont, NMX=8000, DSMAX=0.05, UZR={5: vals1}, STOP=[f'UZ{len(vals1)}'], NPR=100,
                         RL1=3.0, RL0=0.0, bidirectional=True)

# continuations in d
vals = [10.0, 100.0]
c3_sols, c3_cont = a.run(starting_point='UZ1', c='qif', ICP=16, NPAR=n_params, NDIM=n_dim, name='d:1',
                         origin=c2_cont, NMX=8000, DSMAX=0.05, UZR={16: vals}, STOP=[f'UZ{len(vals)}'], NPR=100,
                         RL1=150.0, RL0=0.0, bidirectional=True)
c4_sols, c4_cont = a.run(starting_point='UZ2', c='qif', ICP=16, NPAR=n_params, NDIM=n_dim, name='d:2',
                         origin=c2_cont, NMX=8000, DSMAX=0.05, UZR={16: vals}, STOP=[f'UZ{len(vals)}'], NPR=100,
                         RL1=150.0, RL0=0.0, bidirectional=True)

# main continuations
####################

# continuations in I for d = 10.0
r1_sols, r1_cont = a.run(starting_point='UZ1', c='qif', ICP=8, NPAR=n_params, NDIM=n_dim, name='I_ext:1',
                         origin=c3_cont, NMX=8000, DSMAX=0.05, UZR={}, STOP=[], NPR=100,
                         RL1=150.0, RL0=0.0, bidirectional=True)
a.run(starting_point='LP1', c='qif2', ICP=[5, 8], name='D/I:lp1', origin=r1_cont, NMX=8000, DSMAX=0.05,
      NPR=20, RL1=5.0, RL0=0.0, bidirectional=True)
a.run(starting_point='LP2', c='qif2', ICP=[5, 8], name='D/I:lp2', origin=r1_cont, NMX=8000, DSMAX=0.05,
      NPR=20, RL1=5.0, RL0=0.0, bidirectional=True)

# continuations in I for d = 100.0
r2_sols, r2_cont = a.run(starting_point='UZ2', c='qif', ICP=8, NPAR=n_params, NDIM=n_dim, name='I_ext:2',
                         origin=c3_cont, NMX=8000, DSMAX=0.05, UZR={}, STOP=[], NPR=100,
                         RL1=150.0, RL0=0.0, bidirectional=True)
r3_sols, r3_cont = a.run(starting_point='UZ2', c='qif', ICP=8, NPAR=n_params, NDIM=n_dim, name='I_ext:3',
                         origin=c4_cont, NMX=8000, DSMAX=0.05, UZR={}, STOP=[], NPR=100,
                         RL1=150.0, RL0=0.0, bidirectional=True)
a.run(starting_point='LP1', c='qif2', ICP=[5, 8], name='D/I:lp3', origin=r2_cont, NMX=8000, DSMAX=0.05,
      NPR=20, RL1=5.0, RL0=0.0, bidirectional=True)
a.run(starting_point='LP2', c='qif2', ICP=[5, 8], name='D/I:lp4', origin=r2_cont, NMX=8000, DSMAX=0.05,
      NPR=20, RL1=5.0, RL0=0.0, bidirectional=True)
a.run(starting_point='HB1', c='qif2', ICP=[5, 8], name='D/I:hb1', origin=r3_cont, NMX=8000, DSMAX=0.05,
      NPR=20, RL1=5.0, RL0=0.0, bidirectional=True)

# save results
fname = '../results/rs.pkl'
kwargs = {'deltas': vals1, 'ds': vals}
a.to_file(fname, **kwargs)
