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
t_sols, t_cont = a.run(e='eic', c='ivp', name='t', DS=1e-4, DSMIN=1e-12, EPSL=1e-06, NPR=1000,
                       EPSU=1e-06, EPSS=1e-04, DSMAX=0.1, NMX=50000, UZR={14: 2000.0}, STOP={'UZ1'},
                       NDIM=n_dim, NPAR=n_params)

########################
# bifurcation analysis #
########################

# preparations
##############

# continuation in background input to RS population
c1_sols, c1_cont = a.run(starting_point='UZ1', c='qif', ICP=18, NPAR=n_params, NDIM=n_dim, name='I_rs:1',
                         origin=t_cont, NMX=8000, DSMAX=0.1, UZR={18: [50.0]}, STOP=[f'UZ1'], NPR=100,
                         RL1=100.0, RL0=0.0)

# continuation in Delta_ib
vals = [0.5, 2.5]
c2_sols, c2_cont = a.run(starting_point='UZ1', c='qif', ICP=30, NPAR=n_params, NDIM=n_dim, name='D_ib:1',
                         origin=c1_cont, NMX=8000, DSMAX=0.1, UZR={30: vals}, STOP=[f'UZ2'], NPR=100, RL1=3.0, RL0=0.0,
                         bidirectional=True)

# main continuations
####################

# continuation in background input to GPe for Delta_ib = 0.5
c3_sols, c3_cont = a.run(starting_point='UZ1', c='qif', ICP=36, NPAR=n_params, NDIM=n_dim, name='I_ib:1',
                         origin=c2_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=50, RL1=500.0)

# continuation of limit cycle for Delta_ib = 0.5
c4_sols, c4_cont = a.run(starting_point='HB1', c='qif2b', ICP=[36, 11], NPAR=n_params, NDIM=n_dim, name='I_ib:1:lc',
                         origin=c3_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=['LP3', 'BP1'], NPR=20, RL1=500.0)

# continuation in background input to GPe for Delta_ib = 2.5
c5_sols, c5_cont = a.run(starting_point='UZ2', c='qif', ICP=36, NPAR=n_params, NDIM=n_dim, name='I_ib:2',
                         origin=c2_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=50, RL1=500.0)

# continuation of limit cycle for Delta_ib = 2.5
c6_sols, c6_cont = a.run(starting_point='HB1', c='qif2b', ICP=[36, 11], NPAR=n_params, NDIM=n_dim, name='I_ib:2:lc',
                         origin=c5_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=['LP3', 'BP1'], NPR=20, RL1=500.0)

# 2D continuations
##################

# continuation in Delta_ib and I_ib
a.run(starting_point='HB1', c='qif2', ICP=[30, 36], NPAR=n_params, NDIM=n_dim, name='D_ib/I_ib:hb1', origin=c3_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=10, RL1=10.0, RL0=0.0, bidirectional=True)
a.run(starting_point='LP1', c='qif2', ICP=[30, 36], NPAR=n_params, NDIM=n_dim, name='D_ib/I_ib:lp1', origin=c3_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=10, RL1=10.0, RL0=0.0, bidirectional=True)
a.run(starting_point='LP2', c='qif2', ICP=[30, 36], NPAR=n_params, NDIM=n_dim, name='D_ib/I_ib:lp2', origin=c3_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=10, RL1=10.0, RL0=0.0, bidirectional=True)
a.run(starting_point='LP1', c='qif_lc', ICP=[30, 36], NPAR=n_params, NDIM=n_dim, name='D_ib/I_ib:lc1', origin=c4_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=['BP2'], NPR=10, RL1=10.0, RL0=0.0)

# save results
##############

fname = '../results/eic_ib.pkl'
kwargs = {'deltas': vals}
a.to_file(fname, **kwargs)
