from pycobi import ODESystem
import sys

"""
Bifurcation analysis of the adaptive Izhikevich mean-field model with distributed etas.
"""

# preparations
##############

path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"

# config
n_dim = 5
n_params = 20
a = ODESystem("etas_sfa_global", working_dir="config", auto_dir=auto_dir, init_cont=False)

# initial continuation in time to converge to fixed point
t_sols, t_cont = a.run(c='ivp', name='t', DS=1e-4, DSMIN=1e-10, EPSL=1e-06, NPR=1000, NPAR=n_params, NDIM=n_dim,
                       EPSU=1e-06, EPSS=1e-05, DSMAX=0.1, NMX=50000, UZR={14: 2000.0}, STOP={'UZ1'})

########################
# bifurcation analysis #
########################

# prepare state
###############

# continuation in synaptic strength
c1_sols, c1_cont = a.run(starting_point='UZ1', c='ss', ICP=4, name='g:1', origin=t_cont,
                         UZR={4: [15.0]}, STOP=[f'UZ1'], RL1=50.0, RL0=0.0, NPAR=n_params, NDIM=n_dim)

# continuations in kappa
vals = [30, 90.0, 300.0]
c2_sols, c2_cont = a.run(starting_point='UZ1', ICP=16, name='kappa:1', origin=c1_cont, UZR={16: vals},
                         STOP=[f'UZ{len(vals)}'], RL1=310.0, RL0=0.0, DSMAX=0.5)

# main continuations
####################

RL0 = -40.0
RL1 = 20.0
STOP = ["BP1", "ZH1", "GH3"]

# continuations in I for low heterogeneity
r1_sols, r1_cont = a.run(starting_point='UZ1', ICP=8, name='I_ext:1', origin=c2_cont, UZR={}, STOP=[],
                         RL1=150.0, RL0=-50.0)
a.run(starting_point='LP1', c='2d', ICP=[9, 8], name='b/I:lp1', origin=r1_cont, RL1=RL1, RL0=RL0, bidirectional=True,
      NPAR=n_params, NDIM=n_dim, UZR={}, STOP=STOP)
a.run(starting_point='LP2', c='2d', ICP=[9, 8], name='b/I:lp2', origin=r1_cont, RL1=RL1, RL0=RL0, bidirectional=True,
      NPAR=n_params, NDIM=n_dim, STOP=STOP)
a.run(starting_point='HB1', c='2d', ICP=[9, 8], name='b/I:hb1', origin=r1_cont, RL1=RL1, RL0=RL0, bidirectional=True,
      NPAR=n_params, NDIM=n_dim, STOP=STOP)
a.run(starting_point='HB1', c='2d', ICP=[9, 8], name='b/I:hb2', origin=r1_cont, RL1=RL1, RL0=RL0, bidirectional=True,
      NPAR=n_params, NDIM=n_dim, STOP=STOP)

# continuations in I for intermediate heterogeneity
r2_sols, r2_cont = a.run(starting_point='UZ2', ICP=8, name='I_ext:2', origin=c2_cont, UZR={}, STOP=[],
                         RL1=150.0, RL0=-50.0)
a.run(starting_point='LP1', c='2d', ICP=[9, 8], name='b/I:lp3', origin=r2_cont, RL1=RL1, RL0=RL0, bidirectional=True,
      NPAR=n_params, NDIM=n_dim, STOP=STOP)
a.run(starting_point='LP2', c='2d', ICP=[9, 8], name='b/I:lp4', origin=r2_cont, RL1=RL1, RL0=RL0, bidirectional=True,
      NPAR=n_params, NDIM=n_dim, STOP=STOP)
a.run(starting_point='HB1', c='2d', ICP=[9, 8], name='b/I:hb3', origin=r2_cont, RL1=RL1, RL0=RL0, bidirectional=True,
      NPAR=n_params, NDIM=n_dim, STOP=STOP)
a.run(starting_point='HB2', c='2d', ICP=[9, 8], name='b/I:hb4', origin=r2_cont, RL1=RL1, RL0=RL0, bidirectional=True,
      NPAR=n_params, NDIM=n_dim, STOP=STOP)

# continuations in I for high heterogeneity
r3_sols, r3_cont = a.run(starting_point='UZ3', ICP=8, NPAR=n_params, NDIM=n_dim, name='I_ext:3',
                         origin=c2_cont, UZR={}, STOP=[], RL1=250.0, RL0=-50.0)
a.run(starting_point='HB1', c='2d', ICP=[9, 8], name='b/I:hb5', origin=r3_cont, RL1=RL1, RL0=RL0, bidirectional=True,
      NPAR=n_params, NDIM=n_dim, STOP=STOP)
a.run(starting_point='HB2', c='2d', ICP=[9, 8], name='b/I:hb6', origin=r3_cont, RL1=RL1, RL0=RL0, bidirectional=True,
      NPAR=n_params, NDIM=n_dim, STOP=STOP)

# save results
fname = '../results/etas_sfa_global.pkl'
kwargs = {'deltas': [1.0] + vals}
a.to_file(fname, **kwargs)
