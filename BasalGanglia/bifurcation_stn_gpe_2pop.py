from pyrates.utility.pyauto import PyAuto
import sys
from matplotlib.pyplot import show

"""Bifurcation analysis of STN-GPe model with two GPe populations (arkypallidal and prototypical) and 
gamma-distributed axonal delays and bi-exponential synapses."""

path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"

# config
n_dim = 37
n_params = 25
a = PyAuto("auto_files", auto_dir=auto_dir)
kwargs = dict()
model = 'stn_gpe_2pop'
fname = f'../results/{model}_conts.pkl'

#########################
# initial continuations #
#########################

# continuation in time
t_sols, t_cont = a.run(e='stn_gpe_3pop', c='ivp', ICP=14, NMX=100000, name='t', UZR={14: 1000.0}, STOP={'UZ1'},
                       NDIM=n_dim, NPAR=n_params)
starting_point = 'UZ1'
starting_cont = t_cont

# choose relative strength of GPe-p vs. GPe-a axons
kp_sols, kp_cont = a.run(starting_point=starting_point, c='qif', ICP=23, NPAR=n_params, name='k_p', NDIM=n_dim,
                         RL0=0.8, RL1=2.1, origin=starting_cont, NMX=2000, DSMAX=0.1, UZR={23: [2.0]}, STOP={})

starting_point = 'UZ1'
starting_cont = kp_cont

# choose relative strength of inter- vs. intra-population coupling inside GPe
ki_sols, ki_cont = a.run(starting_point=starting_point, c='qif', ICP=24, NPAR=n_params, name='k_i', NDIM=n_dim,
                         RL0=0.8, RL1=2.1, origin=starting_cont, NMX=2000, DSMAX=0.1, UZR={24: [1.5]}, STOP={})

starting_point = 'UZ1'
starting_cont = ki_cont

# preparation of healthy state
##############################

# continuation of eta_a
c2_b0_sols, c2_b0_cont = a.run(starting_point=starting_point, c='qif', ICP=3, NPAR=n_params,
                               name=f'eta_a:tmp', NDIM=n_dim, RL0=-1.0, RL1=10.0, origin=starting_cont,
                               NMX=8000, DSMAX=0.1, UZR={3: 1.0})
starting_point = 'UZ1'
starting_cont = c2_b0_cont

# continuation of eta_e
c2_b1_sols, c2_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=1, NPAR=n_params,
                               name=f'eta_e:tmp', NDIM=n_dim, RL0=-1.0, RL1=10.0, origin=starting_cont,
                               NMX=8000, DSMAX=0.1, UZR={1: [4.0]})
starting_point = 'UZ1'
starting_cont = c2_b1_cont

# continuation of eta_p
c2_b2_sols, c2_b2_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params,
                               name=f'eta_p:tmp', NDIM=n_dim, RL0=-1.0, RL1=10.0, origin=starting_cont,
                               NMX=4000, DSMAX=0.1, UZR={2: [4.0]})
starting_point = 'UZ1'
starting_cont = c2_b2_cont

# continuation of k_ae
c2_b3_sols, c2_b3_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif', ICP=6, NDIM=n_dim,
                               NPAR=n_params, RL1=10.0, NMX=4000, DSMAX=0.05, name=f'k_ae:tmp',
                               UZR={6: [3.0]})

starting_point = 'UZ1'
starting_cont = c2_b3_cont

# continuation of k_pe
c2_b4_sols, c2_b4_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif', ICP=5, NDIM=n_dim,
                               NPAR=n_params, RL1=25.0, NMX=4000, DSMAX=0.05, name=f'k_pe:tmp',
                               UZR={5: [16.0, 21.0]})

starting_point = 'UZ1'
starting_cont = c2_b4_cont

# de-couple GPe-a from GPe-p
############################

# continuations of k_gp for k_pe = 16.0
c2_b5_sols, c2_b5_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif', ICP=22, NDIM=n_dim,
                               NPAR=n_params, RL0=0.0, RL1=20.0, NMX=12000, DSMAX=0.05, name=f'k_gp:1',
                               bidirectional=True, UZR={22: [5.0]})

# 2D continuation of Hopf curve in eta_p and k_pa
c2_b5_2d1_sols, c2_b5_2d1_cont = a.run(starting_point='HB1', origin=c2_b5_cont, c='qif2', ICP=[10, 2], NDIM=n_dim,
                                       NPAR=n_params, RL0=0.0, RL1=10.0, NMX=20000, DSMAX=0.1,
                                       name=f'k_pp/eta_p:hb1', bidirectional=True, UZR={10: 0.0}, STOP=['UZ1'])

# pd-related continuations
##########################

starting_point = 'UZ1'
starting_cont = c2_b5_2d1_cont

# 2. continuation of k_pe for k_pa = 0.0
c2_b6_sols, c2_b6_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif', ICP=5, NDIM=n_dim,
                               NPAR=n_params, RL1=25.0, NMX=4000, DSMAX=0.05, name=f'k_pe:tmp',
                               UZR={5: [8.0]}, DS='-')

# continuations of k_gp for k_pe = 8.0
c2_b7_sols, c2_b7_cont = a.run(starting_point='UZ1', origin=c2_b6_cont, c='qif', ICP=22, NDIM=n_dim,
                               NPAR=n_params, RL0=0.0, RL1=20.0, NMX=12000, DSMAX=0.05, name=f'k_gp:1',
                               bidirectional=True)

# 2D continuation of k_gp and k_pe
c2_b7_2d1_sols, c2_b7_2d1_cont = a.run(starting_point='HB1', origin=c2_b7_cont, c='qif2', ICP=[5, 22], NDIM=n_dim,
                                       NPAR=n_params, RL0=0.0, RL1=25.0, NMX=20000, DSMAX=0.1,
                                       name=f'k_gp/k_pe:hb1', bidirectional=True, UZR={22: [5.0, 3.98]})
c2_b7_2d2_sols, c2_b7_2d2_cont = a.run(starting_point='HB2', origin=c2_b7_cont, c='qif2', ICP=[5, 22], NDIM=n_dim,
                                       NPAR=n_params, RL0=0.0, RL1=25.0, NMX=20000, DSMAX=0.1,
                                       name=f'k_gp/k_pe:hb2', bidirectional=True)
c2_b8_sols, c2_b8_cont = a.run(starting_point='UZ2', origin=c2_b7_2d1_cont, c='qif', ICP=5, name='k_pe:tmp', DSMAX=0.05,
                               NMX=1000, NPAR=n_params, NDIM=n_dim, RL0=0.0, RL1=1.5, DS='-')
c2_b8_2d1_sols, c2_b8_2d1_cont = a.run(starting_point='HB1', origin=c2_b8_cont, c='qif2', ICP=[5, 22], NDIM=n_dim,
                                       NPAR=n_params, RL0=0.0, RL1=25.0, NMX=20000, DSMAX=0.1,
                                       name=f'k_gp/k_pe:hb3', bidirectional=True)
c2_b9_sols, c2_b9_cont = a.run(starting_point='UZ1', origin=c2_b7_2d1_cont, c='qif', ICP=5, name='k_gp:tmp',
                               DSMAX=0.05, NMX=1000, NPAR=n_params, NDIM=n_dim, RL0=0.0, RL1=1.5, DS='-')
c2_b9_2d1_sols, c2_b9_2d1_cont = a.run(starting_point='LP1', origin=c2_b9_cont, c='qif2', ICP=[5, 22], NDIM=n_dim,
                                       NPAR=n_params, RL0=0.0, RL1=25.0, NMX=20000, DSMAX=0.1,
                                       name=f'k_gp/k_pe:lp1', bidirectional=True)

# save results
a.to_file(fname, **kwargs)
