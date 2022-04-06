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
n_dim = 9
n_params = 42
a = PyAuto("config", auto_dir=auto_dir)

# initial continuation in time (to converge to fixed point)
t_sols, t_cont = a.run(e='stn_gpe', c='ivp', name='t', DS=1e-4, DSMIN=1e-12, EPSL=1e-06, NPR=1000,
                       EPSU=1e-06, EPSS=1e-04, DSMAX=0.1, NMX=50000, UZR={14: 2000.0}, STOP={'UZ1'},
                       NDIM=n_dim, NPAR=n_params)

########################
# bifurcation analysis #
########################

# preparations
##############

# continuation in background input to STN
c1_sols, c1_cont = a.run(starting_point='UZ1', c='qif', ICP=16, NPAR=n_params, NDIM=n_dim, name='I_s:1',
                         origin=t_cont, NMX=8000, DSMAX=0.01, UZR={16: [70.0]}, STOP=[f'UZ1'], NPR=100,
                         RL1=100.0, RL0=0.0)

# continuation in background input to GPe
c2_sols, c2_cont = a.run(starting_point='UZ1', c='qif', ICP=33, NPAR=n_params, NDIM=n_dim, name='I_g:1',
                         origin=c1_cont, NMX=8000, DSMAX=0.01, UZR={33: [60.0]}, STOP=[f'UZ1'], NPR=100,
                         RL1=100.0, RL0=0.0)

# main continuation
###################

# continuation in Delta_g
c3_sols, c3_cont = a.run(starting_point='UZ1', c='qif', ICP=27, NPAR=n_params, NDIM=n_dim, name='D_g:1',
                         origin=c2_cont, NMX=8000, DSMAX=0.01, UZR={27: 1.0}, STOP=[], NPR=100, RL1=6.0, RL0=0.01,
                         bidirectional=True)

# continuation in input to STN
c4_sols, c4_cont = a.run(starting_point='UZ1', c='qif', ICP=16, NPAR=n_params, NDIM=n_dim, name='I_s:2',
                         origin=c2_cont, NMX=8000, DSMAX=0.01, UZR={}, STOP=[], NPR=50, RL1=100.0, RL0=0.0,
                         bidirectional=True)

# continuation of limit cycle
c5_sols, c5_cont = a.run(starting_point='HB1', c='qif2b', ICP=[16, 11], NPAR=n_params, NDIM=n_dim, name='I_s:2:lc',
                         origin=c4_cont, NMX=8000, DSMAX=0.01, UZR={}, STOP=[], NPR=50, RL1=100.0)

# 2D continuations
##################

# continuation in both Deltas
a.run(starting_point='HB1', c='qif2', ICP=[6, 27], NPAR=n_params, NDIM=n_dim, name='D_s/D_g:hb1', origin=c4_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=10, RL1=10.0, RL0=0.01, bidirectional=True)

# continuation in Delta_g and k_gs
a.run(starting_point='HB1', c='qif2', ICP=[42, 27], NPAR=n_params, NDIM=n_dim, name='w/D_g:hb1', origin=c3_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=10, RL1=10.0, RL0=0.0, bidirectional=True)

# continuation in Delta_g and I_s
a.run(starting_point='HB1', c='qif2', ICP=[27, 16], NPAR=n_params, NDIM=n_dim, name='D_g/I_s:hb1', origin=c3_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=10, RL1=10.0, RL0=0.0, bidirectional=True)

# save results
##############

fname = '../results/stn_gpe.pkl'
kwargs = {}
a.to_file(fname, **kwargs)
