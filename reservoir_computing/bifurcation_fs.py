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
a.run(e='fs', c='ivp', name='t', DS=1e-4, DSMIN=1e-10, EPSL=1e-06, NPR=1000, NPAR=n_params, NDIM=n_dim,
      EPSU=1e-06, EPSS=1e-05, DSMAX=0.1, NMX=50000, UZR={14: 500.0}, STOP={'UZ1'})

########################
# bifurcation analysis #
########################

# 1D continuations
##################

NPR = 20

# continuation in background input
a.run(starting_point='UZ1', c='qif', ICP=8, NPAR=n_params, NDIM=n_dim, name='eta:1',
      origin='t', NMX=8000, DSMAX=0.1, UZR={8: [200.0]}, STOP=[], NPR=NPR, RL1=210.0, RL0=0.0)

# continuation in synaptic time constant
taus = [5.0, 10.0, 20.0]
a.run(starting_point='UZ1', c='qif', ICP=17, NPAR=n_params, NDIM=n_dim, name='tau_s:1',
      origin='eta:1', NMX=8000, DSMAX=0.1, UZR={17: taus}, STOP=[], NPR=NPR, RL1=50.0, RL0=0.0, bidirectional=True)

# continuations in coupling strength
for i in range(len(taus)):
    a.run(starting_point=f'UZ{i+1}', c='qif', ICP=4, NPAR=n_params, NDIM=n_dim, name=f'g:{i+1}',
          origin='tau_s:1', NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=NPR, RL1=100.0, RL0=0.0,
          bidirectional=True)

# 2D continuations
##################

NPR = 20

# 2D continuation follow-up in g and eta
a.run(starting_point='HB1', c='qif2', ICP=[4, 8], name='g/eta:hb1', origin=f'g:2', NMX=8000, DSMAX=0.01,
      NPR=NPR, RL1=100.0, RL0=0.0, bidirectional=True)

# 2D continuation follow-up in g and tau_s
a.run(starting_point='HB1', c='qif2', ICP=[4, 17], name='g/tau_s:hb1', origin=f'g:2', NMX=8000, DSMAX=0.01,
      NPR=NPR, RL1=100.0, RL0=0.0, bidirectional=True)

# save results
fname = '../results/fs_bifs.pkl'
kwargs = {"tau_s": taus}
a.to_file(fname, **kwargs)
