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
t_sols, t_cont = a.run(e='eiic2', c='ivp', name='t', DS=1e-4, DSMIN=1e-12, EPSL=1e-06, EPSU=1e-06, EPSS=1e-04,
                       DSMAX=0.1, NMX=50000, UZR={14: 2000.0}, STOP={'UZ1'}, NPR=1000, NDIM=n_dim, NPAR=n_params)

########################
# bifurcation analysis #
########################

# a) preparations
#################

# continuation in background input to LTS population
c0_sols, c0_cont = a.run(starting_point='UZ1', c='qif', ICP=54, NPAR=n_params, NDIM=n_dim, name='I_lts:1',
                         origin=t_cont, NMX=8000, DSMAX=0.1, UZR={54: [110.0]}, STOP=['UZ1'], NPR=100,
                         RL1=150.0)

# continuation in background input to RS population
c1_sols, c1_cont = a.run(starting_point='UZ1', c='qif', ICP=18, NPAR=n_params, NDIM=n_dim, name='I_rs:1',
                         origin=c0_cont, NMX=8000, DSMAX=0.1, UZR={18: [60.0]}, STOP=[], NPR=100,
                         RL1=100.0)

# continuation in Delta_lts
vals = [0.3, 0.6, 1.2]
c2_sols, c2_cont = a.run(starting_point='UZ1', c='qif', ICP=48, NPAR=n_params, NDIM=n_dim, name='D_lts:1',
                         origin=c1_cont, NMX=8000, DSMAX=0.1, UZR={48: vals}, STOP=[], NPR=100, RL1=2.0,
                         RL0=0.0, bidirectional=True)

# b) main 1D continuations
##########################

# continuation in FS input for low Delta_lts
c3_sols, c3_cont = a.run(starting_point='UZ1', c='qif', ICP=36, NPAR=n_params, NDIM=n_dim, name='I_fs:1',
                         origin=c2_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=20, RL1=300.0)
lc0_sols, lc0_cont = a.run(starting_point='HB1', c='qif2b', ICP=[36, 11], NPAR=n_params, NDIM=n_dim, name='I_fs:1:lc1',
                           origin=c3_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=['LP3', 'BP1'], NPR=20, RL1=200.0, RL0=0.0)
a.run(starting_point='HB2', c='qif2b', ICP=[36, 11], NPAR=n_params, NDIM=n_dim, name='I_fs:1:lc2',
      origin=c3_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=['LP3', 'BP1'], NPR=20, RL1=200.0, RL0=0.0)

# continuation in LTS input for intermediate Delta_lts
c4_sols, c4_cont = a.run(starting_point='UZ2', c='qif', ICP=36, NPAR=n_params, NDIM=n_dim, name='I_fs:2',
                         origin=c2_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=20, RL1=300.0)
lc1_sols, lc1_cont = a.run(starting_point='HB1', c='qif2b', ICP=[36, 11], NPAR=n_params, NDIM=n_dim, name='I_fs:2:lc1',
                           origin=c4_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=['LP3', 'BP1'], NPR=20, RL1=200.0, RL0=0.0)
a.run(starting_point='HB3', c='qif2b', ICP=[36, 11], NPAR=n_params, NDIM=n_dim, name='I_fs:2:lc2',
      origin=c4_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=['LP3', 'BP1'], NPR=20, RL1=200.0, RL0=0.0)

# continuation in LTS input for high Delta_lts
c5_sols, c5_cont = a.run(starting_point='UZ3', c='qif', ICP=36, NPAR=n_params, NDIM=n_dim, name='I_fs:3',
                         origin=c2_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=20, RL1=300.0)
a.run(starting_point='HB1', c='qif2b', ICP=[36, 11], NPAR=n_params, NDIM=n_dim, name='I_fs:3:lc1',
      origin=c5_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=['LP3', 'BP1'], NPR=20, RL1=200.0, RL0=0.0)

# c) 2D continuations
#####################

# 2D continuation in Delta_fs and I_fs for low Delta_lts
a.run(starting_point='LP1', c='qif2', ICP=[30, 36], NPAR=n_params, NDIM=n_dim, name='D_fs/I_fs:1:lp1', origin=c3_cont,
      NMX=8000, DSMAX=0.05, UZR={}, STOP=['CP2'], NPR=10, RL1=2.0, RL0=0.0, bidirectional=True)
a.run(starting_point='LP2', c='qif2', ICP=[30, 36], NPAR=n_params, NDIM=n_dim, name='D_fs/I_fs:1:lp2', origin=c3_cont,
      NMX=8000, DSMAX=0.05, UZR={}, STOP=['CP2'], NPR=10, RL1=2.0, RL0=0.0, bidirectional=True)
a.run(starting_point='HB1', c='qif2', ICP=[30, 36], NPAR=n_params, NDIM=n_dim, name='D_fs/I_fs:1:hb1', origin=c3_cont,
      NMX=8000, DSMAX=0.05, UZR={}, STOP=['CP2'], NPR=10, RL1=2.0, RL0=0.0, bidirectional=True)
a.run(starting_point='HB2', c='qif2', ICP=[30, 36], NPAR=n_params, NDIM=n_dim, name='D_fs/I_fs:1:hb2', origin=c3_cont,
      NMX=8000, DSMAX=0.05, UZR={}, STOP=['CP2'], NPR=10, RL1=2.0, RL0=0.0, bidirectional=True)
a.run(starting_point='PD2', c='qif3', ICP=[30, 36, 11], NPAR=n_params, NDIM=n_dim, name='D_fs/I_fs:1:pd1',
      origin=lc0_cont, NMX=2000, DSMAX=0.1, UZR={}, STOP=['BP1', 'LP2'], NPR=10, RL1=10.0, RL0=0.001,
      bidirectional=True)

# 2D continuation in Delta_fs and I_fs for low Delta_lts
a.run(starting_point='LP1', c='qif2', ICP=[30, 36], NPAR=n_params, NDIM=n_dim, name='D_fs/I_fs:2:lp1', origin=c4_cont,
      NMX=8000, DSMAX=0.05, UZR={}, STOP=['CP2'], NPR=10, RL1=2.0, RL0=0.0, bidirectional=True)
a.run(starting_point='LP2', c='qif2', ICP=[30, 36], NPAR=n_params, NDIM=n_dim, name='D_fs/I_fs:2:lp2', origin=c4_cont,
      NMX=8000, DSMAX=0.05, UZR={}, STOP=['CP2'], NPR=10, RL1=2.0, RL0=0.0, bidirectional=True)
a.run(starting_point='HB1', c='qif2', ICP=[30, 36], NPAR=n_params, NDIM=n_dim, name='D_fs/I_fs:2:hb1', origin=c4_cont,
      NMX=8000, DSMAX=0.05, UZR={}, STOP=['CP2'], NPR=10, RL1=2.0, RL0=0.0, bidirectional=True)
a.run(starting_point='HB3', c='qif2', ICP=[30, 36], NPAR=n_params, NDIM=n_dim, name='D_fs/I_fs:2:hb2', origin=c4_cont,
      NMX=8000, DSMAX=0.05, UZR={}, STOP=['CP2'], NPR=10, RL1=2.0, RL0=0.0, bidirectional=True)

# 2D continuation in Delta_fs and I_fs for high Delta_lts
a.run(starting_point='LP1', c='qif2', ICP=[30, 36], NPAR=n_params, NDIM=n_dim, name='D_fs/I_fs:3:lp1', origin=c5_cont,
      NMX=8000, DSMAX=0.05, UZR={}, STOP=['CP2'], NPR=10, RL1=2.0, RL0=0.0, bidirectional=True)
# a.run(starting_point='LP2', c='qif2', ICP=[30, 36], NPAR=n_params, NDIM=n_dim, name='D_fs/I_fs:3:lp2', origin=c5_cont,
#       NMX=8000, DSMAX=0.05, UZR={}, STOP=['CP2'], NPR=10, RL1=2.0, RL0=0.0, bidirectional=True, EPSL=1e-6, EPSU=1e-6,
#       DSMIN=1e-8, EPSS=1e-4)
a.run(starting_point='HB1', c='qif2', ICP=[30, 36], NPAR=n_params, NDIM=n_dim, name='D_fs/I_fs:3:hb1', origin=c5_cont,
      NMX=8000, DSMAX=0.05, UZR={}, STOP=['CP2'], NPR=10, RL1=2.0, RL0=0.0, bidirectional=True)

# save results
##############

fname = '../results/eiic.pkl'
kwargs = {'deltas': vals}
a.to_file(fname, **kwargs)
