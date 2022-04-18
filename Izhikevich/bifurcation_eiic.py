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
n_dim = 15
n_params = 66
a = PyAuto("config", auto_dir=auto_dir)

# initial continuation in time (to converge to fixed point)
t_sols, t_cont = a.run(e='eiic', c='ivp', name='t', DS=1e-4, DSMIN=1e-12, EPSL=1e-06, EPSU=1e-06, EPSS=1e-04,
                       DSMAX=0.1, NMX=50000, UZR={14: 2000.0}, STOP={'UZ1'}, NPR=1000, NDIM=n_dim, NPAR=n_params)

########################
# bifurcation analysis #
########################

# a) analysis in LTS parameters
###############################

# continuation in background input to RS population
c1_sols, c1_cont = a.run(starting_point='UZ1', c='qif', ICP=18, NPAR=n_params, NDIM=n_dim, name='I_rs:1',
                         origin=t_cont, NMX=8000, DSMAX=0.1, UZR={18: [50.0]}, STOP=[f'UZ1'], NPR=100,
                         RL1=100.0)

# continuation in Delta_lts
vals = [0.2, 1.6]
c2_sols, c2_cont = a.run(starting_point='UZ1', c='qif', ICP=48, NPAR=n_params, NDIM=n_dim, name='D_lts:1',
                         origin=c1_cont, NMX=8000, DSMAX=0.1, UZR={48: vals}, STOP=[], NPR=100, RL1=3.0,
                         RL0=0.0, bidirectional=True)

# continuation in LTS input for low Delta_lts
c3_sols, c3_cont = a.run(starting_point='UZ1', c='qif', ICP=54, NPAR=n_params, NDIM=n_dim, name='I_lts:1',
                         origin=c2_cont, NMX=8000, DSMAX=0.1, UZR={54: [105.0]}, STOP=[], NPR=20, RL1=200.0)

# continuation in LTS input for high Delta_lts
c4_sols, c4_cont = a.run(starting_point='UZ2', c='qif', ICP=54, NPAR=n_params, NDIM=n_dim, name='I_lts:2',
                         origin=c2_cont, NMX=8000, DSMAX=0.1, UZR={54: [90.0]}, STOP=[], NPR=20, RL1=200.0)

# 2D continuation in Delta_lts and I_lts
a.run(starting_point='LP1', c='qif2', ICP=[48, 54], NPAR=n_params, NDIM=n_dim, name='D_lts/I_lts:lp1', origin=c3_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=['CP2'], NPR=10, RL1=5.0, RL0=0.0, bidirectional=True)
a.run(starting_point='LP2', c='qif2', ICP=[48, 54], NPAR=n_params, NDIM=n_dim, name='D_lts/I_lts:lp2', origin=c3_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=['CP2'], NPR=10, RL1=5.0, RL0=0.0, bidirectional=True)

# b) analysis in RS parameters
##############################

# continuation in background input to RS population
c5_sols, c5_cont = a.run(starting_point='UZ1', c='qif', ICP=18, NPAR=n_params, NDIM=n_dim, name='I_rs:2',
                         origin=c3_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=20, RL1=200.0, bidirectional=True)
# a.run(starting_point='HB1', c='qif2b', ICP=18, NPAR=n_params, NDIM=n_dim, name='I_rs:2:lc1',
#       origin=c5_cont, NMX=8000, DSMAX=0.05, UZR={}, STOP=[], NPR=20, RL1=200.0)

# 2D continuation in Delta_rs and I_rs
a.run(starting_point='LP1', c='qif2', ICP=[7, 18], NPAR=n_params, NDIM=n_dim, name='D_rs/I_rs:lp1', origin=c5_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=['CP2'], NPR=10, RL1=5.0, RL0=0.0, bidirectional=True)
a.run(starting_point='HB1', c='qif2', ICP=[7, 18], NPAR=n_params, NDIM=n_dim, name='D_rs/I_rs:hb1', origin=c5_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=10, RL1=5.0, RL0=0.0, bidirectional=True)

# 2D continuation in Delta_lts and I_rs
a.run(starting_point='LP1', c='qif2', ICP=[48, 18], NPAR=n_params, NDIM=n_dim, name='D_lts/I_rs:lp1', origin=c5_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=['CP2'], NPR=10, RL1=5.0, RL0=0.0, bidirectional=True)
a.run(starting_point='LP2', c='qif2', ICP=[48, 18], NPAR=n_params, NDIM=n_dim, name='D_lts/I_rs:lp2', origin=c5_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=['CP2'], NPR=10, RL1=5.0, RL0=0.0, bidirectional=True)
a.run(starting_point='HB1', c='qif2', ICP=[48, 18], NPAR=n_params, NDIM=n_dim, name='D_lts/I_rs:hb1', origin=c5_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=10, RL1=5.0, RL0=0.0, bidirectional=True)

# 2D continuation in Delta_fs and I_rs
a.run(starting_point='LP1', c='qif2', ICP=[30, 18], NPAR=n_params, NDIM=n_dim, name='D_fs/I_rs:lp1', origin=c5_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=['CP2'], NPR=10, RL1=5.0, RL0=0.0, bidirectional=True)
a.run(starting_point='LP2', c='qif2', ICP=[30, 18], NPAR=n_params, NDIM=n_dim, name='D_fs/I_rs:lp2', origin=c5_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=['CP2'], NPR=10, RL1=5.0, RL0=0.0, bidirectional=True)
a.run(starting_point='HB1', c='qif2', ICP=[30, 18], NPAR=n_params, NDIM=n_dim, name='D_fs/I_rs:hb1', origin=c5_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=10, RL1=5.0, RL0=0.0, bidirectional=True)

# c) analysis in RS parameters
##############################

# continuation of FS heterogeneity
c6_sols, c6_cont = a.run(starting_point='UZ1', c='qif', ICP=30, NPAR=n_params, NDIM=n_dim, name='D_fs:1',
                         origin=c3_cont, NMX=8000, DSMAX=0.1, UZR={30: [0.5]}, STOP=['UZ1'], NPR=100, RL0=0.0, DS='-')

# continuation in background input to FS population
c7_sols, c7_cont = a.run(starting_point='UZ1', c='qif', ICP=36, NPAR=n_params, NDIM=n_dim, name='I_fs:1',
                         origin=c6_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=20, RL1=200.0)
# a.run(starting_point='HB1', c='qif2b', ICP=18, NPAR=n_params, NDIM=n_dim, name='I_rs:2:lc1',
#       origin=c5_cont, NMX=8000, DSMAX=0.05, UZR={}, STOP=[], NPR=20, RL1=200.0)

# 2D continuation in Delta_rs and I_fs
a.run(starting_point='LP1', c='qif2', ICP=[7, 36], NPAR=n_params, NDIM=n_dim, name='D_rs/I_fs:lp1', origin=c7_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=['CP2'], NPR=10, RL1=5.0, RL0=0.0, bidirectional=True)
a.run(starting_point='HB1', c='qif2', ICP=[7, 36], NPAR=n_params, NDIM=n_dim, name='D_rs/I_fs:hb1', origin=c7_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=10, RL1=5.0, RL0=0.0, bidirectional=True, EPSL=1e-05, EPSU=1e-05,
      EPSS=1e-03)

# 2D continuation in Delta_lts and I_fs
a.run(starting_point='LP1', c='qif2', ICP=[48, 36], NPAR=n_params, NDIM=n_dim, name='D_lts/I_fs:lp1', origin=c7_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=['CP2'], NPR=10, RL1=5.0, RL0=0.0, bidirectional=True)
a.run(starting_point='LP2', c='qif2', ICP=[48, 36], NPAR=n_params, NDIM=n_dim, name='D_lts/I_fs:lp2', origin=c7_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=['CP2'], NPR=10, RL1=5.0, RL0=0.0, bidirectional=True)
a.run(starting_point='HB1', c='qif2', ICP=[48, 36], NPAR=n_params, NDIM=n_dim, name='D_lts/I_fs:hb1', origin=c7_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=10, RL1=5.0, RL0=0.0, bidirectional=True, EPSL=1e-05, EPSU=1e-05,
      EPSS=1e-03)

# 2D continuation in Delta_fs and I_fs
a.run(starting_point='LP1', c='qif2', ICP=[30, 36], NPAR=n_params, NDIM=n_dim, name='D_fs/I_fs:lp1', origin=c7_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=['CP2'], NPR=10, RL1=5.0, RL0=0.0, bidirectional=True)
a.run(starting_point='LP2', c='qif2', ICP=[30, 36], NPAR=n_params, NDIM=n_dim, name='D_fs/I_fs:lp2', origin=c7_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=['CP2'], NPR=10, RL1=5.0, RL0=0.0, bidirectional=True)
a.run(starting_point='HB1', c='qif2', ICP=[30, 36], NPAR=n_params, NDIM=n_dim, name='D_fs/I_fs:hb1', origin=c7_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=10, RL1=5.0, RL0=0.0, bidirectional=True, EPSL=1e-05, EPSU=1e-05,
      EPSS=1e-03)

# save results
##############

fname = '../results/eiic_lts.pkl'
kwargs = {'deltas': vals}
a.to_file(fname, **kwargs)
