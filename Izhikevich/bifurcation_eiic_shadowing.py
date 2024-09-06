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

# continuation in d_rs
vals = [10.0, 100.0]
c2_sols, c2_cont = a.run(starting_point='UZ1', c='qif', ICP=18, NPAR=n_params, NDIM=n_dim, name='d_rs:1',
                         origin=c1_cont, NMX=8000, DSMAX=0.1, UZR={18: vals}, STOP=[], NPR=100, RL1=120.0,
                         RL0=0.0, bidirectional=True)

# continuation in Delta_lts
vals = [0.1, 0.6, 1.8]
c3_sols, c3_cont = a.run(starting_point='UZ1', c='qif', ICP=41, NPAR=n_params, NDIM=n_dim, name='D_lts:1',
                         origin=c2_cont, NMX=8000, DSMAX=0.1, UZR={41: vals}, STOP=[], NPR=100, RL1=2.0,
                         RL0=0.0, bidirectional=True)

# continuation in Delta_lts
vals = [0.1, 0.6, 1.8]
c4_sols, c4_cont = a.run(starting_point='UZ2', c='qif', ICP=41, NPAR=n_params, NDIM=n_dim, name='D_lts:2',
                         origin=c2_cont, NMX=8000, DSMAX=0.1, UZR={41: vals}, STOP=[], NPR=100, RL1=2.0,
                         RL0=0.0, bidirectional=True)

# 2D bifurcation analysis in Delta_rs and I_lts
###############################################

# continuation in I_lts for d_rs = 10.0 and D_lts = 0.1
r1_sols, r1_cont = a.run(starting_point='UZ1', c='qif', ICP=45, NPAR=n_params, NDIM=n_dim, name='I_lts:1',
                         origin=c3_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=100, RL1=200.0)
a.run(starting_point='LP1', c='qif2', ICP=[6, 45], NPAR=n_params, NDIM=n_dim, name='D_rs/I_lts:1:lp1', origin=r1_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=['CP2'], NPR=10, RL1=3.0, RL0=0.0, bidirectional=True, EPSS=1e-6)
a.run(starting_point='LP2', c='qif2', ICP=[6, 45], NPAR=n_params, NDIM=n_dim, name='D_rs/I_lts:1:lp2', origin=r1_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=['CP2'], NPR=10, RL1=3.0, RL0=0.0, bidirectional=True, EPSS=1e-6)
a.run(starting_point='HB1', c='qif2', ICP=[6, 45], NPAR=n_params, NDIM=n_dim, name='D_rs/I_lts:1:hb1', origin=r1_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=['CP2'], NPR=10, RL1=3.0, RL0=0.0, bidirectional=True, EPSS=1e-6)
a.run(starting_point='HB2', c='qif2', ICP=[6, 45], NPAR=n_params, NDIM=n_dim, name='D_rs/I_lts:1:hb2', origin=r1_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=['CP2'], NPR=10, RL1=3.0, RL0=0.0, bidirectional=True, EPSS=1e-6)

# continuation in I_lts for d_rs = 10.0 and D_lts = 0.6
r2_sols, r2_cont = a.run(starting_point='UZ2', c='qif', ICP=45, NPAR=n_params, NDIM=n_dim, name='I_lts:2',
                         origin=c3_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=100, RL1=200.0)
a.run(starting_point='LP1', c='qif2', ICP=[6, 45], NPAR=n_params, NDIM=n_dim, name='D_rs/I_lts:2:lp1', origin=r2_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=['CP2'], NPR=10, RL1=3.0, RL0=0.0, bidirectional=True, EPSS=1e-6)
a.run(starting_point='LP2', c='qif2', ICP=[6, 45], NPAR=n_params, NDIM=n_dim, name='D_rs/I_lts:2:lp2', origin=r2_cont,
      NMX=8000, DSMAX=0.1, UZR={41: [1.5]}, STOP=['CP2'], NPR=10, RL1=3.0, RL0=0.0, bidirectional=True, EPSS=1e-6)
a.run(starting_point='HB1', c='qif2', ICP=[6, 45], NPAR=n_params, NDIM=n_dim, name='D_rs/I_lts:2:hb1', origin=r2_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=['CP2'], NPR=10, RL1=3.0, RL0=0.0, bidirectional=True, EPSS=1e-6)

# continuation in I_lts for d_rs = 10.0 and D_lts = 1.8
r3_sols, r3_cont = a.run(starting_point='UZ3', c='qif', ICP=45, NPAR=n_params, NDIM=n_dim, name='I_lts:3',
                         origin=c3_cont, NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=100, RL1=200.0)
a.run(starting_point='LP1', c='qif2', ICP=[6, 45], NPAR=n_params, NDIM=n_dim, name='D_rs/I_lts:3:lp1', origin=r2_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=['CP2'], NPR=10, RL1=3.0, RL0=0.0, bidirectional=True, EPSS=1e-6)
a.run(starting_point='LP2', c='qif2', ICP=[6, 45], NPAR=n_params, NDIM=n_dim, name='D_rs/I_lts:3:lp2', origin=r2_cont,
      NMX=8000, DSMAX=0.1, UZR={41: [1.5]}, STOP=['CP2'], NPR=10, RL1=3.0, RL0=0.0, bidirectional=True, EPSS=1e-6)

# continuation in I_lts for d_rs = 100.0 and D_lts = 0.1
r4_sols, r4_cont = a.run(starting_point='UZ1', c='qif', ICP=45, NPAR=n_params, NDIM=n_dim, name='I_lts:4',
                         origin=c4_cont, NMX=8000, DSMAX=0.1, UZR={45: 107.0}, STOP=[], NPR=100, RL1=200.0)
a.run(starting_point='UZ1', c='qif', ICP=6, NPAR=n_params, NDIM=n_dim, name='D_rs:1',
      origin=r4_cont, NMX=8000, DSMAX=0.1, UZR={6: 0.15}, STOP=[], NPR=100, RL0=0.1, DS="-")
a.run(starting_point='UZ1', c='qif', ICP=45, NPAR=n_params, NDIM=n_dim, name='I_lts:4:tmp',
      origin='D_rs:1', NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=100, RL0=50.0, RL1=150.0, bidirectional=True)
a.run(starting_point='LP1', c='qif2', ICP=[6, 45], NPAR=n_params, NDIM=n_dim, name='D_rs/I_lts:4:lp1',
      origin='I_lts:4:tmp', NMX=8000, DSMAX=0.1, UZR={}, STOP=['CP2'], NPR=10, RL1=3.0, RL0=0.0, bidirectional=True,
      EPSS=1e-6)
a.run(starting_point='LP2', c='qif2', ICP=[6, 45], NPAR=n_params, NDIM=n_dim, name='D_rs/I_lts:4:lp2',
      origin='I_lts:4:tmp', NMX=8000, DSMAX=0.1, UZR={41: [1.5]}, STOP=['CP2'], NPR=10, RL1=3.0, RL0=0.0,
      bidirectional=True, EPSS=1e-6)
a.run(starting_point='HB1', c='qif2', ICP=[6, 45], NPAR=n_params, NDIM=n_dim, name='D_rs/I_lts:4:hb1', origin=r4_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=['CP2'], NPR=10, RL1=3.0, RL0=0.0, bidirectional=True, EPSS=1e-6)

# continuation in I_lts for d_rs = 100.0 and D_lts = 0.6
r5_sols, r5_cont = a.run(starting_point='UZ2', c='qif', ICP=45, NPAR=n_params, NDIM=n_dim, name='I_lts:5',
                         origin=c4_cont, NMX=8000, DSMAX=0.1, UZR={45: 98.0}, STOP=[], NPR=100, RL1=200.0)
a.run(starting_point='UZ1', c='qif', ICP=6, NPAR=n_params, NDIM=n_dim, name='D_rs:2',
      origin=r5_cont, NMX=8000, DSMAX=0.1, UZR={6: 0.25}, STOP=[], NPR=100, RL0=0.1, DS="-")
a.run(starting_point='UZ1', c='qif', ICP=45, NPAR=n_params, NDIM=n_dim, name='I_lts:5:tmp',
      origin='D_rs:2', NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=100, RL0=50.0, RL1=150.0, bidirectional=True)
a.run(starting_point='LP1', c='qif2', ICP=[6, 45], NPAR=n_params, NDIM=n_dim, name='D_rs/I_lts:5:lp1',
      origin='I_lts:5:tmp', NMX=8000, DSMAX=0.1, UZR={}, STOP=['CP2'], NPR=10, RL1=3.0, RL0=0.0, bidirectional=True,
      EPSS=1e-6)
a.run(starting_point='LP2', c='qif2', ICP=[6, 45], NPAR=n_params, NDIM=n_dim, name='D_rs/I_lts:5:lp2',
      origin='I_lts:5:tmp', NMX=8000, DSMAX=0.1, UZR={41: [1.5]}, STOP=['CP2'], NPR=10, RL1=3.0, RL0=0.0,
      bidirectional=True, EPSS=1e-6)
a.run(starting_point='HB1', c='qif2', ICP=[6, 45], NPAR=n_params, NDIM=n_dim, name='D_rs/I_lts:5:hb1', origin=r5_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=['CP2'], NPR=10, RL1=3.0, RL0=0.0, bidirectional=True, EPSS=1e-6)

# continuation in I_lts for d_rs = 100.0 and D_lts = 1.8
r6_sols, r6_cont = a.run(starting_point='UZ3', c='qif', ICP=45, NPAR=n_params, NDIM=n_dim, name='I_lts:6',
                         origin=c4_cont, NMX=8000, DSMAX=0.1, UZR={45: 80.0}, STOP=[], NPR=100, RL1=200.0)
a.run(starting_point='UZ1', c='qif', ICP=6, NPAR=n_params, NDIM=n_dim, name='D_rs:3',
      origin=r6_cont, NMX=8000, DSMAX=0.1, UZR={6: 0.75}, STOP=[], NPR=100, RL1=1.0)
a.run(starting_point='UZ1', c='qif', ICP=45, NPAR=n_params, NDIM=n_dim, name='I_lts:6:tmp',
      origin='D_rs:3', NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=100, RL0=50.0, RL1=150.0, bidirectional=True)
a.run(starting_point='LP1', c='qif2', ICP=[6, 45], NPAR=n_params, NDIM=n_dim, name='D_rs/I_lts:6:lp1', origin=r6_cont,
      NMX=8000, DSMAX=0.1, UZR={}, STOP=['CP2'], NPR=10, RL1=3.0, RL0=0.0, bidirectional=True, EPSS=1e-6)
a.run(starting_point='LP2', c='qif2', ICP=[6, 45], NPAR=n_params, NDIM=n_dim, name='D_rs/I_lts:6:lp2', origin=r6_cont,
      NMX=8000, DSMAX=0.1, UZR={41: [1.5]}, STOP=['CP2'], NPR=10, RL1=3.0, RL0=0.0, bidirectional=True, EPSS=1e-6)
a.run(starting_point='HB1', c='qif2', ICP=[6, 45], NPAR=n_params, NDIM=n_dim, name='D_rs/I_lts:6:hb1',
      origin='I_lts:6:tmp', NMX=8000, DSMAX=0.1, UZR={}, STOP=['CP2'], NPR=10, RL1=3.0, RL0=0.0, bidirectional=True,
      EPSS=1e-6)

# save results
##############

# fname = '../results/eiic_shadowing.pkl'
# kwargs = {'deltas': vals}
# a.to_file(fname, **kwargs)
