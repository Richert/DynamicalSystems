from pycobi import ODESystem
import sys

"""
Bifurcation analysis of the Fast-Spiking Neuron Izhikevich mean-field model.

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
a = ODESystem("fs", working_dir="config", auto_dir=auto_dir, init_cont=False)

# initial continuation in time (to converge to fixed point)
t_sols, t_cont = a.run(c='ivp', name='t', DS=1e-4, DSMIN=1e-10, EPSL=1e-06, NPR=1000, NDIM=n_dim, NPAR=n_params,
                       EPSU=1e-06, EPSS=1e-05, DSMAX=0.1, NMX=50000, UZR={14: 500.0}, STOP={'UZ1'})

########################
# bifurcation analysis #
########################

# main continuation
###################

# continuation in Delta
vals = [0.5]
c1_sols, c1_cont = a.run(starting_point='UZ1', c='qif', ICP=6, NPAR=n_params, NDIM=n_dim, name='D:1',
                         origin=t_cont, NMX=8000, DSMAX=0.01, UZR={6: vals}, STOP=[f'UZ{len(vals)}'], NPR=100,
                         RL1=10.0, RL0=0.0, bidirectional=True)

for i, v in enumerate(vals):

    # continuation in the input strength
    a.run(starting_point=f'UZ{i+1}', c='qif', ICP=16, NPAR=n_params, NDIM=n_dim, name=f'I:{i+1}', origin=c1_cont,
          NMX=8000, DSMAX=0.1, NPR=20, RL1=700.0)

    # continuation of limit cycle
    a.run(starting_point='HB1', c='qif2b', ICP=16, NPAR=n_params, NDIM=n_dim, name=f'I:{i+1}:lc',
          origin=f'I:{i+1}', NMX=10000, DSMAX=0.1, NPR=20, RL1=131.0, RL0=0.0, STOP=['BP1', 'LP3'])

# 2D continuation follow-up I
target = 0
a.run(starting_point='HB1', c='qif2', ICP=[6, 16], name='D/I:hb1', origin=f'I:{target+1}', NMX=8000, DSMAX=0.1,
      NPR=10, RL1=10.0, RL0=0.001, bidirectional=True)

# save results
fname = '../results/fs.pkl'
kwargs = {'D': vals, 'target': target}
a.to_file(fname, **kwargs)
a.close_session(clear_files=True)
