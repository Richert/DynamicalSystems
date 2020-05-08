from pyauto import PyAuto
import numpy as np
import matplotlib.pyplot as plt

"""Bifurcation analysis of GPe model with two populations (arkypallidal and prototypical) and 
gamma-dstributed axonal delays and bi-exponential synapses."""

# config
n_dim = 18
n_params = 23
a = PyAuto("auto_files")
fname = '../results/gpe_2pop.pkl'

################################
# initial continuation in time #
################################

t_sols, t_cont = a.run(e='gpe_2pop', c='ivp', ICP=14, DS=5e-3, DSMIN=1e-4, DSMAX=1.0, NMX=1000000, name='t',
                       UZR={14: 10000.0}, STOP={'UZ1'}, NDIM=n_dim, NPAR=n_params)

########################################################################
# c3: investigation of GPe behavior for strong GPe-p to GPe-a coupling #
########################################################################

starting_point = 'UZ1'
starting_cont = t_cont

# continuation of GPe intrinsic coupling
########################################

# step 1: codim 1 investigation
s0_sols, s0_cont = a.run(starting_point=starting_point, c='qif', ICP=19, NPAR=n_params, name='k_gp', NDIM=n_dim,
                         RL0=0.99, RL1=100.0, origin=starting_cont, NMX=6000, DSMAX=0.1,
                         UZR={19: [10.0, 20.0, 30.0, 40.0]}, STOP={})

starting_point = 'UZ2'
starting_cont = s0_cont

# step 1: codim 1 investigation
s1_sols, s1_cont = a.run(starting_point=starting_point, c='qif', ICP=20, NPAR=n_params, name='k_p', NDIM=n_dim,
                         RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.1, bidirectional=True,
                         UZR={20: [0.25, 0.5, 2.0, 4.0]}, STOP={})

starting_point = 'UZ2'
starting_cont = s0_cont

# step 1: codim 1 investigation
s2_sols, s2_cont = a.run(starting_point=starting_point, c='qif', ICP=21, NPAR=n_params, name='k_i', NDIM=n_dim,
                         RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.1, bidirectional=True,
                         UZR={21: [0.25, 0.5, 2.0, 4.0]}, STOP={})

starting_point = 'UZ2'
starting_cont = s0_cont

# step 1: codim 1 investigation
s3_sols, s3_cont = a.run(starting_point=starting_point, c='qif', ICP=22, NPAR=n_params, name='k_ap', NDIM=n_dim,
                         RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.1, bidirectional=True,
                         UZR={22: [0.25, 0.5, 2.0, 4.0]}, STOP={})

starting_point = 'UZ3'
starting_cont = s3_cont

# continuation in STR-GPe projection
####################################

# step 1: codim 1 investigation of STR -> GPe
c3_b1_sol, c3_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=18, NPAR=n_params, name='c3:eta_s',
                              NDIM=n_dim, RL0=0.0019, RL1=0.1, origin=starting_cont, NMX=6000, DSMAX=0.1,
                              bidirectional=True)

# continuation in STN -> GPe projection
#######################################

# step 1: codim 1 investigation of STN -> GPe
c3_b2_sols, c3_b2_cont = a.run(starting_point=starting_point, c='qif', ICP=1, NPAR=n_params, name='c3:ta_e', NDIM=n_dim,
                               RL0=0.01, RL1=0.05, origin=starting_cont, NMX=6000, DSMAX=0.1, bidirectional=True)

# continuation of eta_p
#######################

# step 1: codim 1 investigation
c3_b3_sols, c3_b3_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params, name='c3:eta_p', NDIM=n_dim,
                               RL0=-20.0, RL1=15.0, origin=starting_cont, NMX=6000, DSMAX=0.1, bidirectional=True)

# continuation of eta_a
#######################

# step 1: codim 1 investigation
c3_b4_sols, c3_b4_cont = a.run(starting_point=starting_point, c='qif', ICP=3, NPAR=n_params, name='c3:eta_a', NDIM=n_dim,
                               RL0=-20.0, RL1=15.0, origin=starting_cont, NMX=6000, DSMAX=0.1, bidirectional=True)

# continuation of delta_p
#########################

# step 1: codim 1 investigation
c3_b5_sols, c3_b5_cont = a.run(starting_point=starting_point, c='qif', ICP=16, NPAR=n_params, name='c3:delta_p',
                               NDIM=n_dim, RL0=0.01, RL1=1.0, origin=starting_cont, NMX=6000, DSMAX=0.1,
                               bidirectional=True)

# step 2: codim 2 investigation of hopf found in step 1
c3_b5_hb1_sols, c3_b5_hb1_cont = a.run(starting_point='HB1', c='qif2', ICP=[17, 16], NPAR=n_params,
                                       name='c3:delta_p/delta_a', NDIM=n_dim, RL0=0.01, RL1=1.0, origin=c3_b5_cont,
                                       NMX=6000, DSMAX=0.5, bidirectional=True)
c3_b5_hb2_sols, c3_b5_hb2_cont = a.run(starting_point='HB1', c='qif2', ICP=[2, 16], NPAR=n_params,
                                       name='c3:delta_p/eta_p', NDIM=n_dim, RL0=-10.0, RL1=10.0, origin=c3_b5_cont,
                                       NMX=6000, DSMAX=0.5, bidirectional=True, STOP={'CP3'})
c3_b5_hb3_sols, c3_b5_hb3_cont = a.run(starting_point='HB1', c='qif2', ICP=[3, 16], NPAR=n_params,
                                       name='c3:delta_p/eta_a', NDIM=n_dim, RL0=-10.0, RL1=10.0, origin=c3_b5_cont,
                                       NMX=6000, DSMAX=0.5, bidirectional=True)
c3_b5_hb4_sols, c3_b5_hb4_cont = a.run(starting_point='HB1', c='qif2', ICP=[20, 16], NPAR=n_params,
                                       name='c3:delta_p/k_p', NDIM=n_dim, RL0=0.1, RL1=10.0, origin=c3_b5_cont,
                                       NMX=6000, DSMAX=0.5, bidirectional=True)

# step 3: continuation of periodic orbit of hopf from step 1
c3_b5_lc1_sols, c3_b5_lc1_cont = a.run(starting_point='HB1', c='qif2b', ICP=[16, 11], NPAR=n_params,
                                       name='c3:delta_p_lc', NDIM=n_dim, RL0=0.01, RL1=1.0, origin=c3_b5_cont,
                                       NMX=6000, DSMAX=0.5, STOP={'BP1', 'PD1'})

# continuation of delta_a
#########################

# step 1: codim 1 investigation
c3_b6_sols, c3_b6_cont = a.run(starting_point=starting_point, c='qif', ICP=17, NPAR=n_params, name='c3:delta_a',
                               NDIM=n_dim, RL0=0.01, RL1=1.0, origin=starting_cont, NMX=6000, DSMAX=0.1,
                               bidirectional=True)

# save results
kwargs = dict()
a.to_file(fname, **kwargs)

##################################################################
# c1: investigation of GPe behavior for strong GPe-p projections #
##################################################################

starting_point = 'UZ1'
starting_cont = t_cont

# continuation of GPe intrinsic coupling
########################################

# step 1: codim 1 investigation
s0_sols, s0_cont = a.run(starting_point=starting_point, c='qif', ICP=19, NPAR=n_params, name='k_gp', NDIM=n_dim,
                         RL0=0.99, RL1=100.0, origin=starting_cont, NMX=6000, DSMAX=0.1,
                         UZR={19: [10.0, 20.0, 30.0, 40.0]}, STOP={})

starting_point = 'UZ2'
starting_cont = s0_cont

# step 1: codim 1 investigation
s1_sols, s1_cont = a.run(starting_point=starting_point, c='qif', ICP=20, NPAR=n_params, name='k_p', NDIM=n_dim,
                         RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.1, bidirectional=True,
                         UZR={20: [0.25, 0.5, 2.0, 4.0]}, STOP={})

starting_point = 'UZ3'
starting_cont = s1_cont

# step 1: codim 1 investigation
s2_sols, s2_cont = a.run(starting_point=starting_point, c='qif', ICP=21, NPAR=n_params, name='k_i', NDIM=n_dim,
                         RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.1, bidirectional=True,
                         UZR={21: [0.25, 0.5, 2.0, 4.0]}, STOP={})

starting_point = 'UZ3'
starting_cont = s2_cont

# step 1: codim 1 investigation
s3_sols, s3_cont = a.run(starting_point=starting_point, c='qif', ICP=22, NPAR=n_params, name='k_ap', NDIM=n_dim,
                         RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.1, bidirectional=True,
                         UZR={22: [0.25, 0.5, 2.0, 4.0]}, STOP={})

starting_point = 'UZ3'
starting_cont = s2_cont

# continuation in STR-GPe projection
####################################

# step 1: codim 1 investigation of STR -> GPe
c1_b1_sol, c1_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=18, NPAR=n_params, name='c1:eta_s',
                              NDIM=n_dim, RL0=0.0019, RL1=0.1, origin=starting_cont, NMX=6000, DSMAX=0.1,
                              bidirectional=True)

# continuation in STN-GPe projection
#####################################

# step 1: codim 1 investigation of STN -> GPe
c1_b2_sols, c1_b2_cont = a.run(starting_point=starting_point, c='qif', ICP=1, NPAR=n_params, name='c1:eta_e', NDIM=n_dim,
                               RL0=0.01, RL1=0.05, origin=starting_cont, NMX=6000, DSMAX=0.1, bidirectional=True)

# continuation of eta_p
#######################

# step 1: codim 1 investigation
c1_b3_sols, c1_b3_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params, name='c1:eta_p', NDIM=n_dim,
                               RL0=-20.0, RL1=15.0, origin=starting_cont, NMX=6000, DSMAX=0.1, bidirectional=True)

# continuation of eta_a
#######################

# step 1: codim 1 investigation
c1_b4_sols, c1_b4_cont = a.run(starting_point=starting_point, c='qif', ICP=3, NPAR=n_params, name='c1:eta_a', NDIM=n_dim,
                               RL0=-20.0, RL1=15.0, origin=starting_cont, NMX=6000, DSMAX=0.1, bidirectional=True)

# step 2: codim 2 investigation of fold found in step 1
c1_b4_fp1_sols, c1_b4_fp1_cont = a.run(starting_point='LP1', c='qif2', ICP=[2, 3], NPAR=n_params,
                                       name='c1:eta_a/eta_p', NDIM=n_dim, RL0=-10, RL1=10.0, origin=c1_b4_cont,
                                       NMX=6000, DSMAX=0.5, bidirectional=True)
c1_b4_fp2_sols, c1_b4_fp2_cont = a.run(starting_point='LP1', c='qif2', ICP=[19, 3], NPAR=n_params,
                                       name='c1:eta_a/k_gp', NDIM=n_dim, RL0=1.0, RL1=100.0, origin=c1_b4_cont,
                                       NMX=6000, DSMAX=0.5, bidirectional=True, STOP={'CP3', 'LP2'})
c1_b4_fp3_sols, c1_b4_fp3_cont = a.run(starting_point='LP1', c='qif2', ICP=[20, 3], NPAR=n_params,
                                       name='c1:eta_a/k_p', NDIM=n_dim, RL0=0.1, RL1=10.0, origin=c1_b4_cont,
                                       NMX=6000, DSMAX=0.5, bidirectional=True, STOP={'CP3', 'LP2'})
c1_b4_fp4_sols, c1_b4_fp4_cont = a.run(starting_point='LP1', c='qif2', ICP=[21, 3], NPAR=n_params,
                                       name='c1:eta_a/k_i', NDIM=n_dim, RL0=0.1, RL1=10.0, origin=c1_b4_cont,
                                       NMX=6000, DSMAX=0.5, bidirectional=True, STOP={'CP3', 'LP2'})

# continuation of delta_p
#########################

# step 1: codim 1 investigation
c1_b5_sols, c1_b5_cont = a.run(starting_point=starting_point, c='qif', ICP=16, NPAR=n_params, name='c1:delta_p',
                               NDIM=n_dim, RL0=0.01, RL1=1.0, origin=starting_cont, NMX=6000, DSMAX=0.1,
                               bidirectional=True)

# step 2: codim 2 investigation of hopf found in step 1
c1_b5_hb1_sols, c1_b5_hb1_cont = a.run(starting_point='HB1', c='qif2', ICP=[17, 16], NPAR=n_params,
                                       name='c1:delta_p/delta_a', NDIM=n_dim, RL0=0.01, RL1=1.0, origin=c1_b5_cont,
                                       NMX=6000, DSMAX=0.5, bidirectional=True)
c1_b5_hb2_sols, c1_b5_hb2_cont = a.run(starting_point='HB1', c='qif2', ICP=[2, 16], NPAR=n_params,
                                       name='c1:delta_p/eta_p', NDIM=n_dim, RL0=-10.0, RL1=10.0, origin=c1_b5_cont,
                                       NMX=6000, DSMAX=0.5, bidirectional=True, STOP={'CP3'})
c1_b5_hb3_sols, c1_b5_hb3_cont = a.run(starting_point='HB1', c='qif2', ICP=[3, 16], NPAR=n_params,
                                       name='c1:delta_p/eta_a', NDIM=n_dim, RL0=-10.0, RL1=10.0, origin=c1_b5_cont,
                                       NMX=6000, DSMAX=0.5, bidirectional=True)
c1_b5_hb4_sols, c1_b5_hb4_cont = a.run(starting_point='HB1', c='qif2', ICP=[20, 16], NPAR=n_params,
                                       name='c1:delta_p/k_p', NDIM=n_dim, RL0=0.1, RL1=10.0, origin=c1_b5_cont,
                                       NMX=6000, DSMAX=0.5, bidirectional=True)

# step 3: continuation of periodic orbit of hopf from step 1
c1_b5_lc1_sols, c1_b5_lc1_cont = a.run(starting_point='HB1', c='qif2b', ICP=[16, 11], NPAR=n_params,
                                       name='c1:delta_p_lc', NDIM=n_dim, RL0=0.01, RL1=1.0, origin=c1_b5_cont,
                                       NMX=6000, DSMAX=0.5, STOP={'BP1', 'PD1'})

# continuation of delta_a
#########################

# step 1: codim 1 investigation
c1_b6_sols, c1_b6_cont = a.run(starting_point=starting_point, c='qif', ICP=17, NPAR=n_params, name='c1:delta_a',
                               NDIM=n_dim, RL0=0.01, RL1=1.0, origin=starting_cont, NMX=6000, DSMAX=0.1,
                               bidirectional=True)

# save results
kwargs = dict()
a.to_file(fname, **kwargs)

#########################################################################
# c2: investigation of GPe behavior for strong GPe-a <-> GPe-p coupling #
#########################################################################

starting_point = 'UZ1'
starting_cont = t_cont

# continuation of GPe intrinsic coupling
########################################

# step 1: codim 1 investigation
s0_sols, s0_cont = a.run(starting_point=starting_point, c='qif', ICP=19, NPAR=n_params, name='k_gp', NDIM=n_dim,
                         RL0=0.99, RL1=100.0, origin=starting_cont, NMX=6000, DSMAX=0.1,
                         UZR={19: [10.0, 20.0, 30.0, 40.0]}, STOP={})

starting_point = 'UZ2'
starting_cont = s0_cont

# step 1: codim 1 investigation
s1_sols, s1_cont = a.run(starting_point=starting_point, c='qif', ICP=20, NPAR=n_params, name='k_p', NDIM=n_dim,
                         RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.1, bidirectional=True,
                         UZR={20: [0.25, 0.5, 2.0, 4.0]}, STOP={})

starting_point = 'UZ2'
starting_cont = s0_cont

# step 1: codim 1 investigation
s2_sols, s2_cont = a.run(starting_point=starting_point, c='qif', ICP=21, NPAR=n_params, name='k_i', NDIM=n_dim,
                         RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.1, bidirectional=True,
                         UZR={21: [0.25, 0.5, 2.0, 4.0]}, STOP={})

starting_point = 'UZ3'
starting_cont = s2_cont

# step 1: codim 1 investigation
s3_sols, s3_cont = a.run(starting_point=starting_point, c='qif', ICP=22, NPAR=n_params, name='k_ap', NDIM=n_dim,
                         RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.1, bidirectional=True,
                         UZR={22: [0.25, 0.5, 2.0, 4.0]}, STOP={})

starting_point = 'UZ3'
starting_cont = s2_cont

# continuation in STR-GPe projection
####################################

# step 1: codim 1 investigation of STR -> GPe
c2_b1_sol, c2_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=18, NPAR=n_params, name='c2:eta_s',
                              NDIM=n_dim, RL0=0.0019, RL1=0.1, origin=starting_cont, NMX=6000, DSMAX=0.1,
                              bidirectional=True)

# continuation in STN -> GPe projection
#######################################

# step 1: codim 1 investigation of STN -> GPe
c2_b2_sols, c2_b2_cont = a.run(starting_point=starting_point, c='qif', ICP=1, NPAR=n_params, name='c2:ta_e', NDIM=n_dim,
                               RL0=0.01, RL1=0.05, origin=starting_cont, NMX=6000, DSMAX=0.1, bidirectional=True)

# continuation of eta_p
#######################

# step 1: codim 1 investigation
c2_b3_sols, c2_b3_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params, name='c2:eta_p', NDIM=n_dim,
                               RL0=-20.0, RL1=15.0, origin=starting_cont, NMX=6000, DSMAX=0.1, bidirectional=True)

# step 2: codim 2 investigation of fold found in step 1
c2_b3_fp1_sols, c2_b3_fp1_cont = a.run(starting_point='LP1', c='qif2', ICP=[3, 2], NPAR=n_params,
                                       name='c2:eta_p/eta_a', NDIM=n_dim, RL0=-10, RL1=10.0, origin=c2_b3_cont,
                                       NMX=6000, DSMAX=0.5, bidirectional=True)
c2_b3_fp2_sols, c2_b3_fp2_cont = a.run(starting_point='LP1', c='qif2', ICP=[19, 2], NPAR=n_params,
                                       name='c2:eta_p/k_gp', NDIM=n_dim, RL0=1.0, RL1=100.0, origin=c2_b3_cont,
                                       NMX=6000, DSMAX=0.5, bidirectional=True, STOP={'CP3'})
c2_b3_fp3_sols, c2_b3_fp3_cont = a.run(starting_point='LP1', c='qif2', ICP=[20, 2], NPAR=n_params,
                                       name='c2:eta_p/k_p', NDIM=n_dim, RL0=0.1, RL1=10.0, origin=c2_b3_cont,
                                       NMX=6000, DSMAX=0.5, bidirectional=True, STOP={'CP3'})
c2_b3_fp4_sols, c2_b3_fp4_cont = a.run(starting_point='LP1', c='qif2', ICP=[21, 2], NPAR=n_params,
                                       name='c2:eta_p/k_i', NDIM=n_dim, RL0=0.1, RL1=10.0, origin=c2_b3_cont,
                                       NMX=6000, DSMAX=0.5, bidirectional=True, STOP={'CP3'})

# continuation of eta_a
#######################

# step 1: codim 1 investigation
c2_b4_sols, c2_b4_cont = a.run(starting_point=starting_point, c='qif', ICP=3, NPAR=n_params, name='c2:eta_a', NDIM=n_dim,
                               RL0=-20.0, RL1=15.0, origin=starting_cont, NMX=6000, DSMAX=0.1, bidirectional=True)

# continuation of delta_p
#########################

# step 1: codim 1 investigation
c2_b5_sols, c2_b5_cont = a.run(starting_point=starting_point, c='qif', ICP=16, NPAR=n_params, name='c2:delta_p',
                               NDIM=n_dim, RL0=0.01, RL1=1.0, origin=starting_cont, NMX=6000, DSMAX=0.1,
                               bidirectional=True)

# continuation of delta_a
#########################

# step 1: codim 1 investigation
c2_b6_sols, c2_b6_cont = a.run(starting_point=starting_point, c='qif', ICP=17, NPAR=n_params, name='c2:delta_a',
                               NDIM=n_dim, RL0=0.01, RL1=1.0, origin=starting_cont, NMX=6000, DSMAX=0.1,
                               bidirectional=True)

# step 2: codim 2 investigation of fold found in step 1
c2_b6_fp1_sols, c2_b6_fp1_cont = a.run(starting_point='LP1', c='qif2', ICP=[16, 17], NPAR=n_params,
                                       name='c2:delta_a/delta_p', NDIM=n_dim, RL0=0.01, RL1=1.0, origin=c2_b6_cont,
                                       NMX=6000, DSMAX=0.5, bidirectional=True)
c2_b6_fp2_sols, c2_b6_fp2_cont = a.run(starting_point='LP1', c='qif2', ICP=[2, 17], NPAR=n_params,
                                       name='c2:delta_a/eta_p', NDIM=n_dim, RL0=-10.0, RL1=10.0, origin=c2_b6_cont,
                                       NMX=6000, DSMAX=0.5, bidirectional=True, STOP={'CP3'})
c2_b6_fp3_sols, c2_b6_fp3_cont = a.run(starting_point='LP1', c='qif2', ICP=[3, 17], NPAR=n_params,
                                       name='c2:delta_a/eta_a', NDIM=n_dim, RL0=-10.0, RL1=10.0, origin=c2_b6_cont,
                                       NMX=6000, DSMAX=0.5, bidirectional=True, STOP={'CP3'})
c2_b6_fp4_sols, c2_b6_fp4_cont = a.run(starting_point='LP1', c='qif2', ICP=[21, 17], NPAR=n_params,
                                       name='c2:delta_a/k_i', NDIM=n_dim, RL0=0.1, RL1=10.0, origin=c2_b6_cont,
                                       NMX=6000, DSMAX=0.5, bidirectional=True, STOP={'CP3'})

# save results
kwargs = dict()
a.to_file(fname, **kwargs)
