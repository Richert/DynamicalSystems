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
n_dim = 15
n_params = 60
a = ODESystem("config", auto_dir=auto_dir, init_cont=False)

# initial continuation in time (to converge to fixed point)
t_sols, t_cont = a.run(e='eiic_shadowing', c='ivp', name='t', DS=1e-4, DSMIN=1e-12, EPSL=1e-06, EPSU=1e-06, EPSS=1e-04,
                       DSMAX=0.1, NMX=50000, UZR={14: 2000.0}, STOP={'UZ1'}, NPR=1000, NDIM=n_dim, NPAR=n_params)

########################
# bifurcation analysis #
########################

# Set up the conditions
#######################

# continuation in background input to RS population
c1_sols, c1_cont = a.run(starting_point='UZ1', c='qif', ICP=15, NPAR=n_params, NDIM=n_dim, name='I_rs:1',
                         origin=t_cont, NMX=8000, DSMAX=0.1, UZR={15: [60.0]}, STOP=[f'UZ1'], NPR=100,
                         RL1=100.0)

# continuation in Delta_fs
vals = [0.1, 1.0]
c2_sols, c2_cont = a.run(starting_point='UZ1', c='qif', ICP=26, NPAR=n_params, NDIM=n_dim, name='D_fs:1',
                         origin=c1_cont, NMX=8000, DSMAX=0.1, UZR={26: vals}, STOP=[], NPR=100, RL1=3.0,
                         RL0=0.0, bidirectional=True)

# continuation in Delta_rs for Delta_fs = 0.1
vals = [0.1, 1.0]
c3_sols, c3_cont = a.run(starting_point='UZ1', c='qif', ICP=6, NPAR=n_params, NDIM=n_dim, name='D_rs:1',
                         origin=c2_cont, NMX=8000, DSMAX=0.1, UZR={6: vals}, STOP=[], NPR=100, RL1=2.0,
                         RL0=0.0, bidirectional=True)

# continuation in Delta_rs for Delta_fs = 1.0
c4_sols, c4_cont = a.run(starting_point='UZ2', c='qif', ICP=6, NPAR=n_params, NDIM=n_dim, name='D_rs:2',
                         origin=c2_cont, NMX=8000, DSMAX=0.1, UZR={6: vals}, STOP=[], NPR=100, RL1=2.0,
                         RL0=0.0, bidirectional=True)

# 2D bifurcation analysis in Delta_rs and I_lts
###############################################

# continuation in I_lts for Delta_fs = 0.1 and Delta_rs = 0.1
r1_sols, r1_cont = a.run(starting_point='UZ1', c='qif', ICP=45, NPAR=n_params, NDIM=n_dim, name='I_lts:1',
                         origin=c3_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=100, RL1=200.0)
a.run(starting_point='LP1', c='qif2', ICP=[41, 45], NPAR=n_params, NDIM=n_dim, name='D_lts/I_lts:1:lp1', origin=r1_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=['CP2'], NPR=10, RL1=3.0, RL0=0.0, bidirectional=True, EPSS=1e-6)
a.run(starting_point='LP2', c='qif2', ICP=[41, 45], NPAR=n_params, NDIM=n_dim, name='D_lts/I_lts:1:lp2', origin=r1_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=['CP2'], NPR=10, RL1=3.0, RL0=0.0, bidirectional=True, EPSS=1e-6)
# a.run(starting_point='HB3', c='qif2', ICP=[6, 45], NPAR=n_params, NDIM=n_dim, name='D_rs/I_lts:1:hb2', origin=r1_cont,
#       NMX=8000, DSMAX=0.1, UZR={}, STOP=['CP2'], NPR=10, RL1=10.0, RL0=0.0, bidirectional=True, EPSS=1e-6)

# continuation in I_lts for Delta_fs = 0.1 and Delta_lts = 0.1
r2_sols, r2_cont = a.run(starting_point='UZ2', c='qif', ICP=45, NPAR=n_params, NDIM=n_dim, name='I_lts:2',
                         origin=c3_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=100, RL1=200.0)
a.run(starting_point='HB1', c='qif2', ICP=[41, 45], NPAR=n_params, NDIM=n_dim, name='D_lts/I_lts:2:hb1', origin=r2_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=['CP2'], NPR=10, RL1=3.0, RL0=0.0, bidirectional=True, EPSS=1e-6)
# a.run(starting_point='HB2', c='qif2', ICP=[6, 45], NPAR=n_params, NDIM=n_dim, name='D_rs/I_lts:2:hb2', origin=r2_cont,
#       NMX=8000, DSMAX=0.1, UZR={}, STOP=['CP2'], NPR=10, RL1=10.0, RL0=0.0, bidirectional=True, EPSS=1e-6)

# continuation in I_lts for Delta_fs = 1.0 and Delta_lts = 0.1
r3_sols, r3_cont = a.run(starting_point='UZ1', c='qif', ICP=45, NPAR=n_params, NDIM=n_dim, name='I_lts:3',
                         origin=c4_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=100, RL1=200.0)
a.run(starting_point='LP1', c='qif2', ICP=[41, 45], NPAR=n_params, NDIM=n_dim, name='D_lts/I_lts:3:lp1', origin=r3_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=['CP2'], NPR=10, RL1=3.0, RL0=0.0, bidirectional=True, EPSS=1e-6)
a.run(starting_point='LP2', c='qif2', ICP=[41, 45], NPAR=n_params, NDIM=n_dim, name='D_lts/I_lts:3:lp2', origin=r3_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=['CP2'], NPR=10, RL1=3.0, RL0=0.0, bidirectional=True, EPSS=1e-6)

# continuation in I_lts for Delta_fs = 1.0 and Delta_lts = 1.0
r4_sols, r4_cont = a.run(starting_point='UZ2', c='qif', ICP=45, NPAR=n_params, NDIM=n_dim, name='I_lts:4',
                         origin=c4_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=100, RL1=200.0)
a.run(starting_point='HB1', c='qif2', ICP=[41, 45], NPAR=n_params, NDIM=n_dim, name='D_lts/I_lts:4:hb1', origin=r4_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=['CP2'], NPR=10, RL1=3.0, RL0=0.0, bidirectional=True, EPSS=1e-6)

# save results
##############

fname = '../results/eiic_shadowing.pkl'
kwargs = {'deltas': vals}
a.to_file(fname, **kwargs)
