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
a = ODESystem("bs_global", working_dir="config", auto_dir=auto_dir, init_cont=False)

# initial continuation in time to converge to fixed point
t_sols, t_cont = a.run(c='ivp', name='t', DS=1e-4, DSMIN=1e-10, EPSL=1e-06, NPR=1000, NPAR=n_params, NDIM=n_dim,
                       EPSU=1e-06, EPSS=1e-05, DSMAX=0.1, NMX=50000, UZR={14: 500.0}, STOP={'UZ1'})

########################
# bifurcation analysis #
########################

# prepare state
###############

# continuation in synaptic strength
c1_sols, c1_cont = a.run(starting_point='UZ1', c='ss', ICP=4, name='g:1', origin=t_cont, NMX=8000, DSMAX=0.05,
                         UZR={4: [15.0]}, STOP=[f'UZ1'], NPR=20, RL1=50.0, RL0=0.0, NPAR=n_params, NDIM=n_dim)

# continuations in kappa
vals = [0.2, 0.4]
c2_sols, c2_cont = a.run(starting_point='UZ1', ICP=16, name='kappa:1', origin=c1_cont, UZR={16: vals},
                         STOP=[f'UZ{len(vals)}'], RL1=100.0, RL0=0.0, bidirectional=True)

# # continuation in Delta
# vals1 = [3.0, 10.0]
# c3_sols, c3_cont = a.run(starting_point='UZ1', ICP=5, name='D:1', origin=c2_cont, UZR={5: vals1},
#                          STOP=[f'UZ{len(vals1)}'], RL1=11.0, RL0=0.0, bidirectional=True)
# c4_sols, c4_cont = a.run(starting_point='UZ2', ICP=5, name='D:2', origin=c2_cont, UZR={5: vals1},
#                          STOP=[f'UZ{len(vals1)}'], RL1=11.0, RL0=0.0, bidirectional=True)
# c5_sols, c5_cont = a.run(starting_point='UZ2', ICP=5, name='D:2', origin=c2_cont, UZR={5: vals1},
#                          STOP=[f'UZ{len(vals1)}'], RL1=11.0, RL0=0.0, bidirectional=True)
#
# conditions = {"c3": {"UZ1": {}, "UZ2": {}},
#               "c4": {"UZ1": {}, "UZ2": {}}}
# for sols, key in zip([c3_sols, c4_sols], ["c3", "c4"]):
#     for p in sols.index:
#         if sols.loc[p, "bifurcation"][0] == "UZ":
#             uz = int(sols.loc[p, "bifurcation_index"][0])
#             conditions[key][f"UZ{uz}"]["kappa"] = sols.loc[p, "PAR(16)"][0]
#             conditions[key][f"UZ{uz}"]["Delta"] = sols.loc[p, "PAR(5)"][0]
#
# print(f"Parameters of UZ1 of c3: {conditions['c3']['UZ1']}")
# print(f"Parameters of UZ2 of c3: {conditions['c3']['UZ2']}")
# print(f"Parameters of UZ1 of c4: {conditions['c4']['UZ1']}")
# print(f"Parameters of UZ2 of c4: {conditions['c4']['UZ2']}")

# main continuations
####################

# continuations in I for small spike frequency adaptation
r1_sols, r1_cont = a.run(starting_point='UZ1', ICP=8, name='I_ext:1', origin=c1_cont, UZR={}, STOP=[],
                         RL1=150.0, RL0=-100.0)
a.run(starting_point='LP1', c='2d', ICP=[5, 8], name='D/I:lp1', origin=r1_cont, RL1=10.0, RL0=1e-4, bidirectional=True,
      NPAR=n_params, NDIM=n_dim)
a.run(starting_point='LP2', c='2d', ICP=[5, 8], name='D/I:lp2', origin=r1_cont, RL1=10.0, RL0=1e-4, bidirectional=True,
      NPAR=n_params, NDIM=n_dim)

# continuations in I for weak spike frequency adaptation
r2_sols, r2_cont = a.run(starting_point='UZ1', ICP=8, name='I_ext:1', origin=c2_cont, UZR={}, STOP=[],
                         RL1=150.0, RL0=-100.0)
a.run(starting_point='LP1', c='2d', ICP=[5, 8], name='D/I:lp3', origin=r2_cont, RL1=10.0, RL0=1e-4, bidirectional=True,
      NPAR=n_params, NDIM=n_dim)
a.run(starting_point='LP2', c='2d', ICP=[5, 8], name='D/I:lp4', origin=r2_cont, RL1=10.0, RL0=1e-4, bidirectional=True,
      NPAR=n_params, NDIM=n_dim)
a.run(starting_point='HB1', c='2d', ICP=[5, 8], name='D/I:hb1', origin=r2_cont, RL1=10.0, RL0=1e-4, bidirectional=True,
      NPAR=n_params, NDIM=n_dim)
a.run(starting_point='HB2', c='2d', ICP=[5, 8], name='D/I:hb2', origin=r2_cont, RL1=10.0, RL0=1e-4, bidirectional=True,
      NPAR=n_params, NDIM=n_dim)

# continuations in I for strong spike frequency adaptation
r3_sols, r3_cont = a.run(starting_point='UZ2', ICP=8, NPAR=n_params, NDIM=n_dim, name='I_ext:3',
                         origin=c2_cont, UZR={}, STOP=[], RL1=150.0, RL0=-100.0)
a.run(starting_point='HB1', c='2d', ICP=[5, 8], name='D/I:hb3', origin=r3_cont, RL1=10.0, RL0=1e-4, bidirectional=True,
      NPAR=n_params, NDIM=n_dim)
a.run(starting_point='HB2', c='2d', ICP=[5, 8], name='D/I:hb4', origin=r3_cont, RL1=10.0, RL0=1e-4, bidirectional=True,
      NPAR=n_params, NDIM=n_dim)

# save results
fname = '../results/bs_global.pkl'
kwargs = {'kappas': [0.0] + vals}
a.to_file(fname, **kwargs)
