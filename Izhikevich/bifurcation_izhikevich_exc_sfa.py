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
n_params = 20
a = PyAuto("config", auto_dir=auto_dir)

# initial continuation in time (to converge to fixed point)
t_sols, t_cont = a.run(e='izhikevich_exc', c='ivp', name='t', DS=1e-4, DSMIN=1e-10, EPSL=1e-06, NPR=1000,
                       EPSU=1e-06, EPSS=1e-05, DSMAX=0.1, NMX=50000, UZR={14: 500.0}, STOP={'UZ1'})

########################
# bifurcation analysis #
########################

# main continuation
###################

# continuation in synaptic strength
c1_sols, c1_cont = a.run(starting_point='UZ1', c='qif', ICP=4, NPAR=n_params, NDIM=n_dim, name='g:1',
                         origin=t_cont, NMX=8000, DSMAX=0.01, UZR={4: [20.0]}, STOP=[f'UZ1'], NPR=100,
                         RL1=1000.0, RL0=0.0)

# continuation in the SFA strength
c2_sols, c2_cont = a.run(starting_point='UZ1', c='qif', ICP=19, NPAR=n_params, NDIM=n_dim, name='d:1',
                         origin=c1_cont, NMX=8000, DSMAX=0.05, UZR={19: [100.0]}, STOP=[f'UZ1'], NPR=100,
                         RL1=200.0, RL0=0.0)

# continuation in Delta
vals = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
c3_sols, c3_cont = a.run(starting_point='UZ1', c='qif', ICP=6, NPAR=n_params, NDIM=n_dim, name='D:1',
                         origin=c2_cont, NMX=8000, DSMAX=0.01, UZR={6: vals}, STOP=[f'UZ{len(vals)}'], NPR=100,
                         RL1=10.0, RL0=0.0, bidirectional=True)

# continuation in resting membrane potential
for i, v in enumerate(vals):
    a.run(starting_point=f'UZ{i+1}', c='qif', ICP=16, NPAR=n_params, NDIM=n_dim, name=f'I:{i+1}', origin=c3_cont,
          NMX=8000, DSMAX=0.1, NPR=10, RL1=100.0, RL0=-100.0, bidirectional=True)

# limit cycle continuation
target = 2
a.run(starting_point='HB1', c='qif2b', ICP=16, NPAR=n_params, NDIM=n_dim, name=f'I:{target+1}:lc',
      origin=f'I:{target+1}', NMX=10000, DSMAX=0.2, NPR=10, RL1=300.0, RL0=0.0, STOP=['BP1', 'LP3'])

# 2D continuation follow-up I
a.run(starting_point='LP1', c='qif2', ICP=[6, 16], name='D/I:lp1', origin=f'I:{target+1}', NMX=8000, DSMAX=0.05,
      NPR=5, RL1=10.0, RL0=0.0, bidirectional=True)
a.run(starting_point='LP2', c='qif2', ICP=[6, 16], name='D/I:lp2', origin=f'I:{target+1}', NMX=8000, DSMAX=0.05,
      NPR=5, RL1=10.0, RL0=0.0, bidirectional=True)
a.run(starting_point='HB1', c='qif2', ICP=[6, 16], name='D/I:hb1', origin=f'I:{target+1}', NMX=8000, DSMAX=0.05,
      NPR=5, RL1=10.0, RL0=0.0, bidirectional=True)
a.run(starting_point='LP2', c='qif_lc', ICP=[6, 16], name='D/I:lc1', origin=f'I:{target+1}:lc', NMX=8000, DSMAX=0.05,
      NPR=5, RL1=10.0, RL0=0.0, bidirectional=True)

# 2D continuation follow-up I
a.run(starting_point='LP1', c='qif2', ICP=[19, 16], name='d/I:lp1', origin=f'I:{target+1}', NMX=8000, DSMAX=0.05,
      NPR=5, RL1=200.0, RL0=0.0, bidirectional=True)
a.run(starting_point='LP2', c='qif2', ICP=[19, 16], name='d/I:lp2', origin=f'I:{target+1}', NMX=8000, DSMAX=0.05,
      NPR=5, RL1=200.0, RL0=0.0, bidirectional=True)
a.run(starting_point='HB1', c='qif2', ICP=[19, 16], name='d/I:hb1', origin=f'I:{target+1}', NMX=8000, DSMAX=0.05,
      NPR=5, RL1=200.0, RL0=0.0, bidirectional=True)
a.run(starting_point='LP2', c='qif_lc', ICP=[19, 16], name='d/I:lc1', origin=f'I:{target+1}:lc', NMX=8000, DSMAX=0.05,
      NPR=5, RL1=200.0, RL0=0.0, bidirectional=True)

# save results
fname = '../results/izhikevich_exc_sfa.pkl'
kwargs = {'D': vals, 'target': target}
a.to_file(fname, **kwargs)
