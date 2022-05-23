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

# a) analysis in LTS parameters
###############################

# continuation in background input to RS population
c1_sols, c1_cont = a.run(starting_point='UZ1', c='qif', ICP=18, NPAR=n_params, NDIM=n_dim, name='I_rs:1',
                         origin=t_cont, NMX=8000, DSMAX=0.1, UZR={18: [60.0]}, STOP=[], NPR=100,
                         RL1=100.0)

# continuation in Delta_fs
vals = [0.2, 0.6]
c2_sols, c2_cont = a.run(starting_point='UZ1', c='qif', ICP=30, NPAR=n_params, NDIM=n_dim, name='D_fs:1',
                         origin=c1_cont, NMX=8000, DSMAX=0.1, UZR={30: vals}, STOP=[], NPR=100, RL1=3.0,
                         RL0=0.0, bidirectional=True)

# continuation in LTS input for low Delta_fs
c3_sols, c3_cont = a.run(starting_point='UZ1', c='qif', ICP=54, NPAR=n_params, NDIM=n_dim, name='I_lts:1',
                         origin=c2_cont, NMX=8000, DSMAX=0.1, UZR={54: [110.0]}, STOP=[], NPR=20, RL1=300.0)
lc1_sols, lc1_cont = a.run(starting_point='HB1', c='qif2b', ICP=[54, 11], NPAR=n_params, NDIM=n_dim, name='I_lts:1:lc1',
                           origin=c3_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=['LP4', 'BP1'], NPR=20, RL1=200.0, RL0=0.0)

# continuation in LTS input for high Delta_fs
c4_sols, c4_cont = a.run(starting_point='UZ2', c='qif', ICP=54, NPAR=n_params, NDIM=n_dim, name='I_lts:2',
                         origin=c2_cont, NMX=8000, DSMAX=0.1, UZR={54: [110.0]}, STOP=[], NPR=20, RL1=300.0)
lc2_sols, lc2_cont = a.run(starting_point='HB1', c='qif2b', ICP=[54, 11], NPAR=n_params, NDIM=n_dim, name='I_lts:2:lc1',
                           origin=c4_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=['LP4', 'BP1'], NPR=20, RL1=200.0, RL0=0.0)

# continuation in Delta_lts
c5_sols, c5_cont = a.run(starting_point='UZ1', c='qif', ICP=30, NPAR=n_params, NDIM=n_dim, name='D_fs:2',
                         origin=c4_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=100, RL1=3.0,
                         RL0=0.0, bidirectional=True)
lc3_sols, lc3_cont = a.run(starting_point='HB1', c='qif2b', ICP=[30, 11], NPAR=n_params, NDIM=n_dim, name='D_fs:2:lc1',
                           origin=c5_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=['LP4', 'BP1'], NPR=20, RL1=3.0, RL0=0.0)
lc4_sols, lc4_cont = a.run(starting_point='HB2', c='qif2b', ICP=[30, 11], NPAR=n_params, NDIM=n_dim, name='D_fs:2:lc2',
                           origin=c5_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=['LP4', 'BP1'], NPR=20, RL1=3.0, RL0=0.0)

# 2D continuation in Delta_lts and Delta_fs
a.run(starting_point='LP1', c='qif2', ICP=[48, 30], NPAR=n_params, NDIM=n_dim, name='D_lts/D_fs:lp1', origin=c5_cont,
      NMX=8000, DSMAX=0.02, UZR={}, STOP=['CP2'], NPR=10, RL1=2.0, RL0=0.0, bidirectional=True)
a.run(starting_point='LP2', c='qif2', ICP=[48, 30], NPAR=n_params, NDIM=n_dim, name='D_lts/D_fs:lp2', origin=c5_cont,
      NMX=8000, DSMAX=0.02, UZR={}, STOP=['CP2'], NPR=10, RL1=2.0, RL0=0.0, bidirectional=True)
a.run(starting_point='HB1', c='qif2', ICP=[48, 30], NPAR=n_params, NDIM=n_dim, name='D_lts/D_fs:hb1', origin=c5_cont,
      NMX=8000, DSMAX=0.02, UZR={}, STOP=['CP2'], NPR=10, RL1=2.0, RL0=0.0, bidirectional=True)
a.run(starting_point='HB2', c='qif2', ICP=[48, 30], NPAR=n_params, NDIM=n_dim, name='D_lts/D_fs:hb2', origin=c5_cont,
      NMX=8000, DSMAX=0.02, UZR={}, STOP=['CP2'], NPR=10, RL1=2.0, RL0=0.0, bidirectional=True)
a.run(starting_point='PD1', c='qif3', ICP=[48, 30, 11], NPAR=n_params, NDIM=n_dim, name='D_lts/D_fs:pd1',
      origin=lc4_cont, NMX=2000, DSMAX=0.05, UZR={}, STOP=['BP1', 'LP2'], NPR=10, RL1=2.0, RL0=0.001,
      bidirectional=True)
a.run(starting_point='PD3', c='qif3', ICP=[48, 30, 11], NPAR=n_params, NDIM=n_dim, name='D_lts/D_fs:pd2',
      origin=lc4_cont, NMX=2000, DSMAX=0.05, UZR={}, STOP=['BP1', 'LP2'], NPR=10, RL1=2.0, RL0=0.001,
      bidirectional=True)

# 2D continuation in Delta_lts and I_lts for high Delta_fs
a.run(starting_point='LP1', c='qif2', ICP=[48, 54], NPAR=n_params, NDIM=n_dim, name='D_lts/I_lts:lp1', origin=c4_cont,
      NMX=8000, DSMAX=0.05, UZR={}, STOP=['CP2'], NPR=10, RL1=2.0, RL0=0.0, bidirectional=True)
a.run(starting_point='LP2', c='qif2', ICP=[48, 54], NPAR=n_params, NDIM=n_dim, name='D_lts/I_lts:lp2', origin=c4_cont,
      NMX=8000, DSMAX=0.05, UZR={}, STOP=['CP2'], NPR=10, RL1=2.0, RL0=0.0, bidirectional=True)
a.run(starting_point='HB1', c='qif2', ICP=[48, 54], NPAR=n_params, NDIM=n_dim, name='D_lts/I_lts:hb1', origin=c4_cont,
      NMX=8000, DSMAX=0.05, UZR={}, STOP=['CP2'], NPR=10, RL1=2.0, RL0=0.0, bidirectional=True)
# a.run(starting_point='LP2', c='qif3', ICP=[48, 54, 11], NPAR=n_params, NDIM=n_dim, name='D_lts/I_lts:lc1',
#       origin=lc2_cont, NMX=2000, DSMAX=0.2, UZR={}, STOP=['BP1', 'LP2'], NPR=10, RL1=10.0, RL0=0.001,
#       bidirectional=True)

# save results
##############

fname = '../results/eiic.pkl'
kwargs = {'deltas': vals}
a.to_file(fname, **kwargs)
