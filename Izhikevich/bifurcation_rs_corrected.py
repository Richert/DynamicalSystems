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

# initial continuation in time to converge to fixed point
t_sols, t_cont = a.run(e='rs_corrected', c='ivp', name='t', DS=1e-4, DSMIN=1e-10, EPSL=1e-05, NPR=1000, NPAR=n_params,
                       NDIM=n_dim, EPSU=1e-05, EPSS=1e-04, DSMAX=0.1, NMX=50000, UZR={14: 500.0}, STOP={'UZ1'})

########################
# bifurcation analysis #
########################

# main continuation
###################

# continuation in synaptic strength
c1_sols, c1_cont = a.run(starting_point='UZ1', c='qif', ICP=4, NPAR=n_params, NDIM=n_dim, name='g:1',
                         origin=t_cont, NMX=8000, DSMAX=0.01, UZR={4: [15.0]}, STOP=[f'UZ1'], NPR=100,
                         RL1=1000.0, RL0=0.0)

# continuation in v_z
vals = [-120, -100.0, -80.0, -60.0]
c2_sols, c2_cont = a.run(starting_point='UZ1', c='qif', ICP=8, NPAR=n_params, NDIM=n_dim, name='v_z:1',
                         origin=c1_cont, NMX=8000, DSMAX=0.1, UZR={8: vals}, STOP=[f'UZ{len(vals)}'], NPR=100,
                         RL1=0.0, RL0=-200.0, bidirectional=True)

# continuation in resting membrane potential
for i, v in enumerate(vals):
    a.run(starting_point=f'UZ{i+1}', c='qif', ICP=16, NPAR=n_params, NDIM=n_dim, name=f'I:{i+1}', origin=c2_cont,
          NMX=8000, DSMAX=0.1, NPR=10, RL1=100.0, RL0=-300.0, bidirectional=True)

# 2D continuation follow-up in g
target = 0
a.run(starting_point='LP1', c='qif2', ICP=[8, 16], name='v_0/I:lp1', origin=f'I:{target+1}', NMX=8000, DSMAX=0.1,
      NPR=10, RL1=-60.0, RL0=-300.0, bidirectional=True)
a.run(starting_point='LP2', c='qif2', ICP=[8, 16], name='v_0/I:lp2', origin=f'I:{target+1}', NMX=8000, DSMAX=0.1,
      NPR=10, RL1=-60.0, RL0=-300.0, bidirectional=True, EPSL=1e-3, EPSS=1e-3, EPSU=1e-3, DSMIN=1e-8)

# save results
fname = '../results/rs_corrected.pkl'
kwargs = {'v_reset': vals}
a.to_file(fname, **kwargs)
