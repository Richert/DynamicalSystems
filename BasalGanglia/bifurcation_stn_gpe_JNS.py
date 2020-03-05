from pyauto import PyAuto
import numpy as np
import matplotlib.pyplot as plt

#####################################
# bifurcation analysis of qif model #
#####################################

# config
n_dim = 30
n_params = 53
a = PyAuto("auto_files")

# initial continuation in time
##############################

t_sols, t_cont = a.run(e='stn_gpe_30d', c='ivp', ICP=14, DS=5e-2, DSMIN=1e-4, DSMAX=1.0, NMX=1000000, name='t',
                       UZR={14: 1000.0}, STOP={'UZ1'}, NDIM=n_dim, NPAR=n_params)

starting_point = 'UZ1'
starting_cont = t_cont

# continuation in k_str
#######################

# step 1: codim 1 investigation
c0_sols, c0_cont = a.run(starting_point=starting_point, c='qif', ICP=32, NPAR=n_params, name='k_str', NDIM=n_dim,
                         RL0=100.0, RL1=1000.0, bidirectional=True, NMX=6000, DSMAX=0.5, STOP={}, UZR={},
                         origin=starting_cont)

# continuation in k
###################

# step 1: codim 1 investigation
c1_sols, c1_cont = a.run(starting_point='UZ1', c='qif', ICP=52, NPAR=n_params, name='k', NDIM=n_dim,
                         RL0=1.0, RL1=2.0, bidirectional=False, NMX=6000, DSMAX=0.05, STOP={}, UZR={},
                         origin=starting_cont)

# step 2: continue HB from step 1 in time constants of system
c1_hb_sols, c1_hb_cont = a.run(starting_point='HB1', c='qif2', ICP=[6, 52, 11], DSMAX=0.05, NMX=4000,
                               NPAR=n_params, name='k/tau_e', origin=c1_cont, NDIM=n_dim, RL0=1.0,
                               RL1=20.0, bidirectional=True)
c1_hb_sols2, c1_hb_cont2 = a.run(starting_point='HB1', c='qif2', ICP=[33, 52, 11], DSMAX=0.05, NMX=4000,
                                 NPAR=n_params, name='k/tau_i', origin=c1_cont, NDIM=n_dim, RL0=10.0,
                                 RL1=30.0, bidirectional=True)

# # continuation in k_i
# ######################
#
# # step 1: codim 1 investigation
# c2_sols, c2_cont = a.run(starting_point='UZ1', c='qif', ICP=53, NPAR=n_params, name='k_i', NDIM=n_dim,
#                          RL0=0.5, RL1=2.0, bidirectional=True, NMX=6000, DSMAX=0.05, STOP={}, UZR={},
#                          origin=starting_cont)

# continuation in delta_i
#########################

# step 1: codim 1 investigation
c3_sols, c3_cont = a.run(starting_point='UZ1', c='qif', ICP=38, NPAR=n_params, name='delta_i', NDIM=n_dim,
                         RL0=0.1, RL1=15.0, DS='-', NMX=6000, DSMAX=0.05, STOP={}, UZR={},
                         origin=starting_cont)

# step 2: continue HB from step 1 in time constants of system
c3_hb_sols, c3_hb_cont = a.run(starting_point='HB1', c='qif2', ICP=[6, 38, 11], DSMAX=0.05, NMX=4000,
                               NPAR=n_params, name='k/tau_e', origin=c1_cont, NDIM=n_dim, RL0=1.0,
                               RL1=20.0, bidirectional=True)
c3_hb_sols2, c3_hb_cont2 = a.run(starting_point='HB1', c='qif2', ICP=[33, 38, 11], DSMAX=0.05, NMX=4000,
                                 NPAR=n_params, name='k/tau_i', origin=c1_cont, NDIM=n_dim, RL0=10.0,
                                 RL1=30.0, bidirectional=True)

# # continuation in delta_e
# #########################
#
# # step 1: codim 1 investigation
# c4_sols, c4_cont = a.run(starting_point='UZ1', c='qif', ICP=15, NPAR=n_params, name='delta_e', NDIM=n_dim,
#                          RL0=0.1, RL1=15.0, DS='-', NMX=6000, DSMAX=0.05, STOP={}, UZR={},
#                          origin=starting_cont)

# save results
fname = '../results/stn_gpe_JNS.pkl'
kwargs = dict()
a.to_file(fname, **kwargs)
