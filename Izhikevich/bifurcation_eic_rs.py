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
n_dim = 10
n_params = 45
a = PyAuto("config", auto_dir=auto_dir)

# initial continuation in time (to converge to fixed point)
t_sols, t_cont = a.run(e='eic', c='ivp', name='t', DS=1e-4, DSMIN=1e-12, EPSL=1e-06, EPSU=1e-06, EPSS=1e-04,
                       DSMAX=0.1, NMX=50000, UZR={14: 2000.0}, STOP={'UZ1'}, NPR=1000, NDIM=n_dim, NPAR=n_params)

########################
# bifurcation analysis #
########################

# preparations
##############

# continuation in background input to IB population
c1_sols, c1_cont = a.run(starting_point='UZ1', c='qif', ICP=36, NPAR=n_params, NDIM=n_dim, name='I_ib:1',
                         origin=t_cont, NMX=8000, DSMAX=0.1, UZR={36: [220.0]}, STOP=[f'UZ1'], NPR=100,
                         RL1=400.0, RL0=0.0)

# continuation in Delta_rs
vals = [0.5, 2.5]
c2_sols, c2_cont = a.run(starting_point='UZ1', c='qif', ICP=7, NPAR=n_params, NDIM=n_dim, name='D_rs:1',
                         origin=c1_cont, NMX=8000, DSMAX=0.1, UZR={7: vals}, STOP=[f'UZ2'], NPR=100, RL1=6.0,
                         RL0=0.0, bidirectional=True)

# main continuations
####################

# continuation in background input to RS for low Delta_rs
c3_sols, c3_cont = a.run(starting_point='UZ1', c='qif', ICP=18, NPAR=n_params, NDIM=n_dim, name='I_rs:1',
                         origin=c2_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=20, RL1=500.0)

# continuation of limit cycles for low Delta_rs
c4_sols, c4_cont = a.run(starting_point='HB1', c='qif2b', ICP=[18, 11], NPAR=n_params, NDIM=n_dim, name='I_rs:1:lc1',
                         origin=c3_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=['LP3', 'BP1'], NPR=20, RL1=500.0)
# c5_sols, c5_cont = a.run(starting_point='HB2', c='qif2b', ICP=[18, 11], NPAR=n_params, NDIM=n_dim, name='I_rs:1:lc2',
#                          origin=c3_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=['LP3', 'BP1'], NPR=20, RL1=500.0)
# c6_sols, c6_cont = a.run(starting_point='HB3', c='qif2b', ICP=[18, 11], NPAR=n_params, NDIM=n_dim, name='I_rs:1:lc3',
#                          origin=c3_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=['LP3', 'BP1'], NPR=20, RL1=500.0)

# continuation in background input to RS for high Delta_rs
c7_sols, c7_cont = a.run(starting_point='UZ2', c='qif', ICP=18, NPAR=n_params, NDIM=n_dim, name='I_rs:2',
                         origin=c2_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=20, RL1=500.0)

# 2D continuations
##################

# continuation in Delta_ib and I_ib
a.run(starting_point='LP1', c='qif2', ICP=[7, 18], NPAR=n_params, NDIM=n_dim, name='D_rs/I_rs:lp1', origin=c3_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=10, RL1=10.0, RL0=0.0, bidirectional=True)
a.run(starting_point='LP2', c='qif2', ICP=[7, 18], NPAR=n_params, NDIM=n_dim, name='D_rs/I_rs:lp2', origin=c3_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=10, RL1=10.0, RL0=0.0, bidirectional=True)
a.run(starting_point='HB1', c='qif2', ICP=[7, 18], NPAR=n_params, NDIM=n_dim, name='D_rs/I_rs:hb1', origin=c3_cont,
      NMX=10000, DSMAX=0.1, UZR={}, STOP=[], NPR=10, RL1=10.0, RL0=-5.0, bidirectional=True, EPSL=1e-05, EPSU=1e-05,
      EPSS=1e-03)
# a.run(starting_point='HB3', c='qif2', ICP=[7, 18], NPAR=n_params, NDIM=n_dim, name='D_rs/I_rs:hb3', origin=c3_cont,
#       NMX=10000, DSMAX=0.1, UZR={}, STOP=[], NPR=10, RL1=10.0, RL0=-5.0, bidirectional=True, EPSL=1e-05, EPSU=1e-05,
#       EPSS=1e-03)

# save results
##############

fname = '../results/eic_rs.pkl'
kwargs = {'deltas': vals}
a.to_file(fname, **kwargs)
