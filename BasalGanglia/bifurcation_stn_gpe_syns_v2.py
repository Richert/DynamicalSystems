from pyauto import PyAuto
import numpy as np
import matplotlib.pyplot as plt

"""Bifurcation analysis of STN-GPe model with gamma-dstributed axonal delays and bi-exponential synapses.
Populations include the STN (exc), GPe-p (inh) and GPe-a (inh)."""

# config
n_dim = 37
n_params = 25
a = PyAuto("auto_files")
fname = '../results/stn_gpe_syns_v2.pkl'

################################
# initial continuation in time #
################################

t_sols, t_cont = a.run(e='stn_gpe_syns_v2', c='ivp', ICP=14, DS=5e-3, DSMIN=1e-4, DSMAX=1.0, NMX=1000000, name='t',
                       UZR={14: 10000.0}, STOP={'UZ1'}, NDIM=n_dim, NPAR=n_params)

#################################
# investigation of GPe coupling #
#################################

starting_point = 'UZ1'
starting_cont = t_cont

# continuation of all GPe afferents
###################################

# step 1: codim 1 investigation
c0_sols, c0_cont = a.run(starting_point=starting_point, c='qif', ICP=21, NPAR=n_params, name='k_gp', NDIM=n_dim,
                         RL0=0.99, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.2)

# # step 2: codim 2 investigation of branch point found in step 1
# c0_2d1_sols, c0_2d1_cont = a.run(starting_point='HB1', c='qif2', ICP=[23, 21], NMX=4000, DSMAX=0.5,
#                                  NPAR=n_params, name='k_gp/k_gp_intra', origin=c0_cont, NDIM=n_dim,
#                                  bidirectional=True, RL0=0.1, RL1=10.0)
# c0_2d2_sols, c0_2d2_cont = a.run(starting_point='HB1', c='qif2', ICP=[24, 21], NMX=4000, DSMAX=0.5,
#                                  NPAR=n_params, name='k_gp/k_gp_inh', origin=c0_cont, NDIM=n_dim,
#                                  bidirectional=True, RL0=0.1, RL1=10.0)

# step 3: codim 1 investigation of periodic orbit found in step 1
c0_lc1_sols, c0_lc1_cont = a.run(starting_point='HB1', c='qif2b', ICP=[21, 11], NMX=4000, DSMAX=1.0,
                                 NPAR=n_params, name='k_gp_lc', origin=c0_cont, NDIM=n_dim,
                                 RL0=0.99, RL1=10.0, STOP={'PD1', 'BP1'})

# save results
kwargs = dict()
a.to_file(fname, **kwargs)

# continuation in GPe-a <-> GPe-p coupling
##########################################

# step 1: codim 1 investigation
c1_sols, c1_cont = a.run(starting_point=starting_point, c='qif', ICP=23, NPAR=n_params, name='k_gp_intra', NDIM=n_dim,
                         RL0=0.99, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.05, STOP={'UZ6'},
                         bidirectional=True)

# # step 2: codim 2 investigation of branch point found in step 1
# c1_2d1_sols, c1_2d1_cont = a.run(starting_point='HB1', c='qif2', ICP=[9, 23], NMX=4000, DSMAX=0.5,
#                                  NPAR=n_params, name='k_gp_intra/k_ap', origin=c1_cont, NDIM=n_dim,
#                                  bidirectional=True, RL0=1.0, RL1=50.0)
# c1_2d2_sols, c1_2d2_cont = a.run(starting_point='HB1', c='qif2', ICP=[10, 23], NMX=4000, DSMAX=0.5,
#                                  NPAR=n_params, name='k_gp_intra/k_pa', origin=c1_cont, NDIM=n_dim,
#                                  bidirectional=True, RL0=20.0, RL1=200.0)
# c1_2d3_sols, c1_2d3_cont = a.run(starting_point='HB1', c='qif2', ICP=[15, 23], NMX=4000, DSMAX=0.5,
#                                  NPAR=n_params, name='k_gp_intra/k_aa', origin=c1_cont, NDIM=n_dim,
#                                  bidirectional=True, RL0=20.0, RL1=200.0)
# c1_2d4_sols, c1_2d4_cont = a.run(starting_point='HB1', c='qif2', ICP=[8, 23], NMX=4000, DSMAX=0.5,
#                                  NPAR=n_params, name='k_gp_intra/k_pp', origin=c1_cont, NDIM=n_dim,
#                                  bidirectional=True, RL0=5.0, RL1=100.0)
#
# # step 3: codim 1 investigation of periodic orbit found in step 1
# c1_lc1_sols, c1_lc1_cont = a.run(starting_point='HB1', c='qif2b', ICP=[23, 11], NMX=4000, DSMAX=1.0,
#                                  NPAR=n_params, name='k_gp_intra_lc', origin=c1_cont, NDIM=n_dim,
#                                  RL0=0.99, RL1=10.0, STOP={'PD1', 'BP1', 'LP4'})

# save results
kwargs = dict()
a.to_file(fname, **kwargs)

# continuation in STR-GPe coupling
##################################

# step 1: codim 1 investigation of STR -> GPe
c2_sols, c2_cont = a.run(starting_point=starting_point, c='qif', ICP=22, NPAR=n_params, name='str', NDIM=n_dim,
                         RL0=0.0019, RL1=0.1, origin=starting_cont, NMX=6000, DSMAX=0.05, STOP={'UZ6'})

# step 2: codim 1 investigation of STR -> GPe-a
c3_sols, c3_cont = a.run(starting_point=starting_point, c='qif', ICP=17, NPAR=n_params, name='k_as', NDIM=n_dim,
                         RL0=399.0, RL1=5000.0, origin=starting_cont, NMX=6000, DSMAX=1.0, STOP={'UZ6'})

# step 3: codim 1 investigation of STR -> GPe-p
c4_sols, c4_cont = a.run(starting_point=starting_point, c='qif', ICP=16, NPAR=n_params, name='k_ps', NDIM=n_dim,
                         RL0=99.0, RL1=5000.0, origin=starting_cont, NMX=6000, DSMAX=1.0, STOP={'UZ6'})

# step 3: codim 1 investigation of periodic orbit found in step 1
c4_lc1_sols, c4_lc1_cont = a.run(starting_point='HB1', c='qif2b', ICP=[16, 11], NMX=4000, DSMAX=5.0,
                                 NPAR=n_params, name='k_ps_lc', origin=c4_cont, NDIM=n_dim,
                                 RL0=99.0, RL1=5000.0, STOP={'PD1', 'BP1'})

# save results
kwargs = dict()
a.to_file(fname, **kwargs)

###########################################
# investigation of firing rate statistics #
###########################################

starting_point = 'UZ1'
starting_cont = t_cont

# continuation in delta_a
#########################

# step 1: codim 1 investigation
c5_sols, c5_cont = a.run(starting_point=starting_point, c='qif', ICP=20, NPAR=n_params, name='delta_a', NDIM=n_dim,
                         RL0=1e-4, RL1=0.3, origin=starting_cont, NMX=6000, DSMAX=0.05, STOP={'UZ6'},
                         bidirectional=True)

# step 2: codim 2 investigation of branch point found in step 1
c5_2d1_sols, c5_2d1_cont = a.run(starting_point='HB1', c='qif2', ICP=[19, 20], NMX=4000, DSMAX=0.5,
                                 NPAR=n_params, name='delta_a/delta_p', origin=c5_cont, NDIM=n_dim,
                                 bidirectional=True, RL0=1e-4, RL1=0.5)

# step 2: codim 1 investigation of periodic orbit found in step 1
c5_lc1_sols, c5_lc1_cont = a.run(starting_point='HB1', c='qif2b', ICP=[20, 11], NMX=4000, DSMAX=1.0,
                                 NPAR=n_params, name='delta_a_lc', origin=c5_cont, NDIM=n_dim,
                                 RL0=1e-4, RL1=0.3, STOP={'PD1', 'BP1'})

# continuation in eta_e
#######################

# step 1: codim 1 investigation
c6_sols, c6_cont = a.run(starting_point=starting_point, c='qif', ICP=1, NPAR=n_params, name='eta_e', NDIM=n_dim,
                         RL0=0.0, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.05, STOP={'UZ6'},
                         bidirectional=True)

# step 3: codim 1 investigation of periodic orbit found in step 1
c6_lc1_sols, c6_lc1_cont = a.run(starting_point='HB1', c='qif2b', ICP=[1, 11], NMX=4000, DSMAX=1.0,
                                 NPAR=n_params, name='eta_e_lc', origin=c6_cont, NDIM=n_dim,
                                 RL0=0.0, RL1=10.0, STOP={'PD1', 'BP1'})

# save results
kwargs = dict()
a.to_file(fname, **kwargs)
