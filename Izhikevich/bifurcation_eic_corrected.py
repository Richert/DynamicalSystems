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
n_dim = 10
n_params = 45
a = ODESystem("config", auto_dir=auto_dir, init_cont=False)

# initial continuation in time (to converge to fixed point)
t_sols, t_cont = a.run(e='eic', c='ivp', name='t', DS=1e-4, DSMIN=1e-12, EPSL=1e-05, EPSU=1e-05, EPSS=1e-04,
                       DSMAX=0.1, NMX=50000, UZR={14: 2000.0}, STOP={'UZ1'}, NPR=1000, NDIM=n_dim, NPAR=n_params)

########################
# bifurcation analysis #
########################

# a) RS-based analysis
######################

# continuation in background input to FS population
c1_sols, c1_cont = a.run(starting_point='UZ1', c='qif', ICP=18, NPAR=n_params, NDIM=n_dim, name='I_rs:1',
                         origin=t_cont, NMX=8000, DSMAX=0.1, UZR={18: [50.0]}, STOP=[f'UZ1'], NPR=100,
                         RL1=100.0)

# continuation in Delta_fs
vals = [0.1, 0.3, 0.6]
c2_sols, c2_cont = a.run(starting_point='UZ1', c='qif', ICP=30, NPAR=n_params, NDIM=n_dim, name='D_fs:1',
                         origin=c1_cont, NMX=8000, DSMAX=0.1, UZR={30: vals}, STOP=[], NPR=100, RL1=3.0,
                         RL0=0.0, bidirectional=True)

# continuation in background input to FS for different Delta_fs
c3_sols, c3_cont = a.run(starting_point='UZ1', c='qif', ICP=36, NPAR=n_params, NDIM=n_dim, name='I_fs:1',
                         origin=c2_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=20, RL1=200.0)
c4_sols, c4_cont = a.run(starting_point='UZ2', c='qif', ICP=36, NPAR=n_params, NDIM=n_dim, name='I_fs:2',
                         origin=c2_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=20, RL1=200.0)
c5_sols, c5_cont = a.run(starting_point='UZ3', c='qif', ICP=36, NPAR=n_params, NDIM=n_dim, name='I_fs:3',
                         origin=c2_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=20, RL1=200.0)

# continuation of limit cycles for different Delta_rs
c6_sols, c6_cont = a.run(starting_point='HB1', c='qif2b', ICP=36, NPAR=n_params, NDIM=n_dim, name='I_fs:1:lc1',
                         origin=c3_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=['LP3', 'BP1'], NPR=20, RL1=200.0, RL0=0.0)
c7_sols, c7_cont = a.run(starting_point='HB1', c='qif2b', ICP=36, NPAR=n_params, NDIM=n_dim, name='I_fs:2:lc1',
                         origin=c4_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=['LP3', 'BP1'], NPR=20, RL1=200.0)
# c8_sols, c8_cont = a.run(starting_point='HB1', c='qif2b', ICP=36, NPAR=n_params, NDIM=n_dim, name='I_fs:3:lc1',
#                          origin=c5_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=['LP3', 'BP1'], NPR=20, RL1=200.0)

# 2D continuation in Delta_fs and I_fs
a.run(starting_point='LP1', c='qif2', ICP=[30, 36], NPAR=n_params, NDIM=n_dim, name='D_fs/I_fs:lp1', origin=c4_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=['CP1'], NPR=10, RL1=10.0, RL0=0.0, bidirectional=True)
a.run(starting_point='LP2', c='qif2', ICP=[30, 36], NPAR=n_params, NDIM=n_dim, name='D_fs/I_fs:lp2', origin=c4_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=['CP1'], NPR=10, RL1=10.0, RL0=0.0, bidirectional=True)
a.run(starting_point='HB1', c='qif2', ICP=[30, 36], NPAR=n_params, NDIM=n_dim, name='D_fs/I_fs:hb1', origin=c4_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=10, RL1=10.0, RL0=-0.2, bidirectional=True, EPSL=1e-06, EPSU=1e-06,
      EPSS=1e-04)
a.run(starting_point='LP1', c='qif_lc', ICP=[30, 36], NPAR=n_params, NDIM=n_dim, name='D_fs/I_fs:lc1', origin=c6_cont,
      NMX=4000, DSMAX=0.05, UZR={}, STOP=['BP1'], NPR=10, RL1=10.0, RL0=-0.2, bidirectional=True, EPSL=1e-06,
      EPSU=1e-06, EPSS=1e-04)
a.run(starting_point='LP2', c='qif_lc', ICP=[30, 36], NPAR=n_params, NDIM=n_dim, name='D_fs/I_fs:lc2', origin=c6_cont,
      NMX=4000, DSMAX=0.05, UZR={}, STOP=['BP1'], NPR=10, RL1=10.0, RL0=-0.2, bidirectional=True, EPSL=1e-06,
      EPSU=1e-06, EPSS=1e-04)

# save results
##############

fname = '../results/eic_corrected2.pkl'
kwargs = {'deltas': vals}
a.to_file(fname, **kwargs)
