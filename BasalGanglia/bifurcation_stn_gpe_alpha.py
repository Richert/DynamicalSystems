from pyauto import PyAuto
import numpy as np
import matplotlib.pyplot as plt

#####################################
# bifurcation analysis of qif model #
#####################################

# config
n_dim = 8
n_params = 10
a = PyAuto("auto_files")

# initial continuation in time
##############################

t_sols, t_cont = a.run(e='stn_gpe_alpha_v2', c='ivp', ICP=14, DS=5e-2, DSMIN=1e-4, DSMAX=1.0, NMX=1000000, name='t',
                       UZR={14: 20000.0}, STOP={'UZ1'}, NDIM=n_dim, NPAR=n_params)

starting_point = 'UZ1'
starting_cont = t_cont

# continuation in k
###################

# step 1: codim 1 investigation
ks = [1.1, 1.2, 1.4, 1.8]
c0_sols, c0_cont = a.run(starting_point=starting_point, c='qif', ICP=10, NPAR=n_params, name='k', NDIM=n_dim,
                         RL0=1.0, RL1=2.0, origin=starting_cont, NMX=4000, DSMAX=0.005, STOP={}, UZR={10: ks})

starting_point = 'UZ1'
starting_cont = c0_cont

# continuation in eta_i
#######################

# step 1: codim 1 investigation
etas = [3., 0., -3.0, -6.0]
c1_sols, c1_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params, name='eta', NDIM=n_dim, DS='-',
                         RL0=-20.0, RL1=20.0, origin=starting_cont, NMX=4000, DSMAX=0.05, STOP={}, UZR={2: etas})

starting_point = 'UZ3'
starting_cont = c1_cont

# continuation in alpha
#######################

# step 1: codim 1 investigation
c2_sols, c2_cont = a.run(starting_point=starting_point, c='qif', ICP=9, NPAR=n_params, name='alpha', NDIM=n_dim,
                         RL0=0.0, RL1=0.2, origin=starting_cont, NMX=4000, DSMAX=0.005, STOP={})

# step 2: codim 2 investigation of limit cycle from step1
c2_d2_sols, c2_d2_cont = a.run(starting_point='HB1', c='qif2', ICP=[10, 9], NPAR=n_params, name='k/alpha', NDIM=n_dim,
                               RL0=1.0, RL1=2.0, origin=c2_cont, NMX=6000, DSMAX=0.05, STOP={}, bidirectional=True,
                               NPR=100)

# step 3: codim 2 investigation of limit cycle from step1
c2_lc_sols, c2_lc_cont = a.run(starting_point='HB1', c='qif2b', ICP=[9, 11], NPAR=n_params, name='alpha_lc', NDIM=n_dim,
                               RL0=0.0, RL1=0.2, origin=c2_cont, NMX=4000, DSMAX=0.05, STOP={'BP2'})

# save results
fname = '../results/stn_gpe_alpha.pkl'
kwargs = {'ks': ks, 'etas': etas}
a.to_file(fname, **kwargs)
