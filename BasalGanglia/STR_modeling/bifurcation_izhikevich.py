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
t_sols, t_cont = a.run(e='izhikevich_fre', c='ivp', name='t', DS=1e-4, DSMIN=1e-10, EPSL=1e-07, NPR=1000,
                       EPSU=1e-07, EPSS=1e-06, DSMAX=1e-2, NMX=50000, UZR={14: 300.0}, STOP={'UZ1'})

########################
# bifurcation analysis #
########################

# main continuation
###################

# continuation in recurrent coupling strength
c1_sols, c1_cont = a.run(starting_point='UZ1', c='qif', ICP=4, NPAR=n_params, NDIM=n_dim, name='g:1',
                         origin=t_cont, NMX=8000, DSMAX=0.05, UZR={4: [20.0]}, STOP=['UZ1'], NPR=100,
                         RL1=100.0, RL0=0.0)

# continuation in background input
c2_sols, c2_cont = a.run(starting_point='UZ1', c='qif', ICP=16, NPAR=n_params, NDIM=n_dim, name='I:1',
                         origin=c1_cont, NMX=8000, DSMAX=0.5, NPR=100, RL1=1000.0, RL0=0.0)

# save results
fname = '../results/izhikevich_fre.pkl'
kwargs = {}
a.to_file(fname, **kwargs)
