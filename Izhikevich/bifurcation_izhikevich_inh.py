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
t_sols, t_cont = a.run(e='izhikevich_inh', c='ivp', name='t', DS=1e-4, DSMIN=1e-10, EPSL=1e-06, NPR=1000,
                       EPSU=1e-06, EPSS=1e-05, DSMAX=0.1, NMX=50000, UZR={14: 500.0}, STOP={'UZ1'})

########################
# bifurcation analysis #
########################

# main continuation
###################

# continuation in synaptic strength
vals = [8.0, 12.0, 16.0, 20.0, 24.0]
c1_sols, c1_cont = a.run(starting_point='UZ1', c='qif', ICP=4, NPAR=n_params, NDIM=n_dim, name='g:1',
                         origin=t_cont, NMX=8000, DSMAX=0.01, UZR={4: vals}, STOP=[f'UZ{len(vals)}'], NPR=100,
                         RL1=1000.0, RL0=0.0)

# continuation in resting membrane potential
for i, v in enumerate(vals):
    a.run(starting_point=f'UZ{i+1}', c='qif', ICP=16, NPAR=n_params, NDIM=n_dim, name=f'I:{i+1}', origin=c1_cont,
          NMX=8000, DSMAX=0.1, NPR=50, RL1=100.0, RL0=0.0, bidirectional=True)

# continuation of limit cycle
target = 3
a.run(starting_point='HB1', c='qif2b', ICP=16, NPAR=n_params, NDIM=n_dim, name=f'I:{target+1}:lc',
      origin=f'I:{target+1}', NMX=10000, DSMAX=0.1, NPR=50, RL1=100.0, RL0=0.0)

# 2D continuation follow-up I
a.run(starting_point='HB1', c='qif2', ICP=[6, 16], name='D/I:hb1', origin=f'I:{target+1}', NMX=8000, DSMAX=0.05,
      NPR=50, RL1=5.0, RL0=0.0, bidirectional=True)
a.run(starting_point='LP1', c='qif2', ICP=[6, 16], name='D/I:lp1', origin=f'I:{target+1}', NMX=8000, DSMAX=0.05,
      NPR=50, RL1=5.0, RL0=0.0, bidirectional=True)
a.run(starting_point='LP2', c='qif2', ICP=[6, 16], name='D/I:lp2', origin=f'I:{target+1}', NMX=8000, DSMAX=0.05,
      NPR=50, RL1=5.0, RL0=0.0, bidirectional=True)

# 2D continuation follow-up II
a.run(starting_point='HB1', c='qif2', ICP=[4, 16], name='g/I:hb1', origin=f'I:{target+1}', NMX=8000, DSMAX=0.05,
      NPR=50, RL1=50.0, RL0=0.0, bidirectional=True)
a.run(starting_point='LP1', c='qif2', ICP=[4, 16], name='g/I:lp1', origin=f'I:{target+1}', NMX=8000, DSMAX=0.05,
      NPR=50, RL1=50.0, RL0=0.0, bidirectional=True)
a.run(starting_point='LP2', c='qif2', ICP=[4, 16], name='g/I:lp2', origin=f'I:{target+1}', NMX=8000, DSMAX=0.05,
      NPR=50, RL1=50.0, RL0=0.0, bidirectional=True)

# save results
fname = '../results/izhikevich_inh.pkl'
kwargs = {'g': vals, 'target': target}
a.to_file(fname, **kwargs)
