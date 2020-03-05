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
                       UZR={14: 10000.0}, STOP={'UZ1'}, NDIM=n_dim, NPAR=n_params)

starting_point = 'UZ1'
starting_cont = t_cont

# continuation in alpha
#######################

# step 1: codim 1 investigation
alphas = [0.01, 0.02, 0.03, 0.04, 0.05]
c0_sols, c0_cont = a.run(starting_point=starting_point, c='qif', ICP=9, NPAR=n_params, name='alpha', NDIM=n_dim,
                         RL0=0.0, RL1=1.0, origin=starting_cont, NMX=1000, DSMAX=0.05, STOP={},
                         UZR={9: alphas})

# step 2: codim 2 investigation of limit cycle from step1
c0_d2_sols, c0_d2_cont = a.run(starting_point='HB1', c='qif2', ICP=[10, 9], NPAR=n_params, name='alpha/beta', NDIM=n_dim,
                               RL0=0.0, RL1=10.0, origin=c0_cont, NMX=6000, DSMAX=0.05, STOP={}, bidirectional=True,
                               NPR=100)

# step 3: codim 2 investigation of limit cycle from step1
c0_lc_sols, c0_lc_cont = a.run(starting_point='HB1', c='qif2b', ICP=[9, 11], NPAR=n_params, name='alpha_lc', NDIM=n_dim,
                               RL0=0.0, RL1=1.0, origin=c0_cont, NMX=4000, DSMAX=0.005, STOP={})

# save results
fname = '../results/stn_gpe_alpha.pkl'
kwargs = {'alphas': alphas}
a.to_file(fname, **kwargs)
