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
a = ODESystem("config", auto_dir=auto_dir, init_cont=False)

# initial continuation in time to converge to fixed point
t_sols, t_cont = a.run(e='rs2', c='ivp', name='t', DS=1e-4, DSMIN=1e-10, EPSL=1e-06, NPR=1000, NPAR=n_params, NDIM=n_dim,
                       EPSU=1e-06, EPSS=1e-05, DSMAX=0.1, NMX=50000, UZR={14: 500.0}, STOP={'UZ1'})

########################
# bifurcation analysis #
########################

# main continuation
###################

# continuation in synaptic strength
c1_sols, c1_cont = a.run(starting_point='UZ1', c='qif', ICP=4, NPAR=n_params, NDIM=n_dim, name='g:1',
                         origin=t_cont, NMX=8000, DSMAX=0.01, UZR={4: [15.0]}, STOP=[f'UZ1'], NPR=100,
                         RL1=1000.0, RL0=0.0)

# continuation in Delta
vals = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
c2_sols, c2_cont = a.run(starting_point='UZ1', c='qif', ICP=6, NPAR=n_params, NDIM=n_dim, name='D:1',
                         origin=c1_cont, NMX=8000, DSMAX=0.01, UZR={6: vals}, STOP=[f'UZ{len(vals)}'], NPR=100,
                         RL1=10.0, RL0=0.0, bidirectional=True)

# continuation in extrinsic input
for i, v in enumerate(vals):
    a.run(starting_point=f'UZ{i+1}', c='qif', ICP=16, NPAR=n_params, NDIM=n_dim, name=f'I:{i+1}', origin=c2_cont,
          NMX=8000, DSMAX=0.1, NPR=10, RL1=100.0)

# 2D continuation follow-up I
target = 2
a.run(starting_point='LP1', c='qif2', ICP=[6, 16], name='D/I:lp1', origin=f'I:{target+1}', NMX=8000, DSMAX=0.05,
      NPR=5, RL1=10.0, RL0=0.0, bidirectional=True)
a.run(starting_point='LP2', c='qif2', ICP=[6, 16], name='D/I:lp2', origin=f'I:{target+1}', NMX=8000, DSMAX=0.05,
      NPR=5, RL1=10.0, RL0=0.0, bidirectional=True)

# save results
fname = '../results/rs.pkl'
kwargs = {'D': vals, 'target': target}
a.to_file(fname, **kwargs)
