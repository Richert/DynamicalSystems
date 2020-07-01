from pyrates.utility.pyauto import PyAuto
import numpy as np
import matplotlib.pyplot as plt

"""Bifurcation analysis of STN-GPe model with two GPe populations (arkypallidal and prototypical) and 
gamma-dstributed axonal delays and bi-exponential synapses."""

# config
n_dim = 37
n_params = 29
a = PyAuto("auto_files")
fname = '../results/stn_gpe_syns.pkl'

# choice of conditions to run bifurcation analysis for
c1 = [  # weak GPe-p <-> GPe-a coupling
      False,  # STN -> GPe-p < STN -> GPe-a
      False,   # STN -> GPe-p > STN -> GPe-a
]
c2 = [  # balanced GPe-p <-> GPe-a coupling
    False,  # STN -> GPe-p < STN -> GPe-a
    False,  # STN -> GPe-p > STN -> GPe-a
]
c3 = [  # strong GPe-p <-> GPe-a coupling
    False,  # STN -> GPe-p < STN -> GPe-a
    True,  # STN -> GPe-p > STN -> GPe-a
]

#########################
# initial continuations #
#########################

# continuation in time
######################

t_sols, t_cont = a.run(e='stn_gpe_syns', c='ivp', ICP=14, NMX=1000000, name='t', UZR={14: 10000.0}, STOP={'UZ1'},
                       NDIM=n_dim, NPAR=n_params)

starting_point = 'UZ1'
starting_cont = t_cont

# continuation of GPe intrinsic coupling
########################################

# step 1: choose base level of GPe coupling strength
s0_sols, s0_cont = a.run(starting_point=starting_point, c='qif', ICP=22, NPAR=n_params, name='k_gp', NDIM=n_dim,
                         RL0=0.99, RL1=100.0, origin=starting_cont, NMX=6000, DSMAX=0.2,
                         UZR={22: [10.0, 20.0, 30.0, 40.0]}, STOP={})

starting_point = 'UZ2'
starting_cont = s0_cont

# step 2: choose relative projection strength of GPe-p vs. GPe-a
s1_sols, s1_cont = a.run(starting_point=starting_point, c='qif', ICP=23, NPAR=n_params, name='k_p', NDIM=n_dim,
                         RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.2, bidirectional=True,
                         UZR={23: [0.5, 0.75, 1.5, 2.0]}, STOP={})

starting_point = 'UZ3'
starting_cont = s1_cont

# step 3: choose relative strength of inter- vs. intra-population coupling inside GPe
s2_sols, s2_cont = a.run(starting_point=starting_point, c='qif', ICP=24, NPAR=n_params, name='k_i', NDIM=n_dim,
                         RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.2, bidirectional=True,
                         UZR={24: [0.5, 0.75, 1.5, 2.0]}, STOP={})

###########################################################################
# c1: investigation of GPe behavior for weak inter-population connections #
###########################################################################

if any(c1):

    starting_point = 'UZ2'
    starting_cont = s2_cont

    # step 4: choose balance between STN -> GPe-p vs. STN -> GPe-a projection
    s3_sols, s3_cont = a.run(starting_point=starting_point, c='qif', ICP=26, NPAR=n_params, name='c1:k_gp_e',
                             NDIM=n_dim, RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.2,
                             bidirectional=True, UZR={26: [0.5, 0.75, 1.5, 2.0]}, STOP={})

    # investigate hopf bifurcation that emerges from asymmetric STN projections
    ###########################################################################

    a.run(starting_point='HB1', origin=s3_cont, c='qif2', ICP=[24, 26], NPAR=n_params, NDIM=n_dim, RL0=0.1, RL1=20.0,
          name='c1:k_gp_e/k_i', NMX=4000, DSMAX=0.5, bidirectional=True)
    a.run(starting_point='HB1', origin=s3_cont, c='qif2', ICP=[8, 26], NPAR=n_params, NDIM=n_dim, RL0=0.0, RL1=20.0,
          name='c1:k_gp_e/k_pp', NMX=4000, DSMAX=0.5, bidirectional=True)
    a.run(starting_point='HB1', origin=s3_cont, c='qif2', ICP=[9, 26], NPAR=n_params, NDIM=n_dim, RL0=0.0, RL1=20.0,
          name='c1:k_gp_e/k_ap', NMX=4000, DSMAX=0.5, bidirectional=True)
    a.run(starting_point='HB1', origin=s3_cont, c='qif2', ICP=[8, 9], NPAR=n_params, NDIM=n_dim, RL0=0.0, RL1=20.0,
          name='c1:k_ap/k_pp', NMX=4000, DSMAX=0.5, bidirectional=True)

    a.run(starting_point='HB1', origin=s3_cont, c='qif2b', ICP=[26, 11], NPAR=n_params, NDIM=n_dim, RL1=10.0,
          name='c1:k_gp_e_lc', NMX=2000, DSMAX=0.2)

    #####################################
    # c1.1: STN -> GPe-p < STN -> GPe-a #
    #####################################

    if c1[0]:

        starting_point = 'UZ2'
        starting_cont = s3_cont

        # continuation of eta_e
        #######################

        # step 1: codim 1 investigation
        c1_b0_sols, c1_b0_cont = a.run(starting_point=starting_point, c='qif', ICP=1, NPAR=n_params,
                                       name='c1:eta_e', NDIM=n_dim, RL0=-10.0, RL1=10.0, origin=starting_cont,
                                       NMX=6000, DSMAX=0.1, bidirectional=True, UZR={1: [-2.0]})

        starting_point = 'UZ1'
        starting_cont = c1_b0_cont

        # continuation of eta_p
        #######################

        # step 1: codim 1 investigation
        c1_b1_sols, c1_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params,
                                       name='c1:eta_p', NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=starting_cont,
                                       NMX=6000, DSMAX=0.1, bidirectional=True, UZR={2: [6.0]})

        starting_point = 'UZ1'
        starting_cont = c1_b1_cont

        # continuation of eta_a
        #######################

        # step 1: codim 1 investigation
        c1_b2_sols, c1_b2_cont = a.run(starting_point=starting_point, c='qif', ICP=3, NPAR=n_params,
                                       name=f'c1:eta_a', NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=starting_cont,
                                       NMX=6000, DSMAX=0.1, bidirectional=True, UZR={3: [0.0]})

        #starting_point = 'UZ1'
        #starting_cont = c1_b2_cont

        # continuation of k_gp
        ######################

        # step 1: codim 1 investigation
        c1_b3_sols, c1_b3_cont = a.run(starting_point=starting_point, c='qif', ICP=22, NPAR=n_params,
                                       name=f'c1:k_gp', NDIM=n_dim, RL1=100.0,
                                       origin=starting_cont, NMX=6000, DSMAX=0.1)

        # continuation of k_ap
        ######################

        # step 1: codim 1 investigation
        c1_b4_sols, c1_b4_cont = a.run(starting_point=starting_point, c='qif', ICP=9, NPAR=n_params,
                                       name=f'c1:k_ap', NDIM=n_dim, RL1=10.0,
                                       origin=starting_cont, NMX=6000, DSMAX=0.1)

        # step 2: codim 2 investigation of hopf bifurcation found in step 1
        a.run(starting_point='HB1', c='qif2', ICP=[24, 9], NPAR=n_params, name=f'c1:k_i/k_ap',
              NDIM=n_dim, RL0=0.1, RL1=10.0, origin=c1_b4_cont, NMX=4000, DSMAX=0.1, bidirectional=True)
        a.run(starting_point='HB1', c='qif2', ICP=[26, 9], NPAR=n_params, name=f'c1:k_gpe_e/k_ap',
              NDIM=n_dim, RL0=0.1, RL1=10.0, origin=c1_b4_cont, NMX=4000, DSMAX=0.1, bidirectional=True)
        a.run(starting_point='HB1', c='qif2', ICP=[8, 9], NPAR=n_params, name=f'c1:k_pp/k_ap',
              NDIM=n_dim, RL0=0.1, RL1=10.0, origin=c1_b4_cont, NMX=4000, DSMAX=0.1, bidirectional=True)
        a.run(starting_point='HB1', c='qif2', ICP=[22, 9], NPAR=n_params, name=f'c1:k_gp/k_ap',
              NDIM=n_dim, RL0=1.0, RL1=100.0, origin=c1_b4_cont, NMX=4000, DSMAX=0.1, bidirectional=True)

        # step 2: continuation of limit cycle found in step 1
        a.run(starting_point='HB1', c='qif2b', ICP=[9, 11], NPAR=n_params, name=f'c1:k_ap_lc',
              NDIM=n_dim, RL1=10.0, origin=c1_b4_cont, NMX=2000, DSMAX=0.2)

        # save results
        kwargs = dict()
        a.to_file(fname, **kwargs)

    #####################################
    # c1.2: STN -> GPe-p > STN -> GPe-a #
    #####################################

    if c1[1]:

        starting_point = 'UZ3'
        starting_cont = s3_cont

        # continuation of eta_e
        #######################

        # step 1: codim 1 investigation
        c1_b0_sols, c1_b0_cont = a.run(starting_point=starting_point, c='qif', ICP=1, NPAR=n_params,
                                       name='c1.2:eta_e', NDIM=n_dim, RL0=-10.0, RL1=10.0, origin=starting_cont,
                                       NMX=6000, DSMAX=0.1, bidirectional=True, UZR={1: [-2.0]})

        starting_point = 'UZ1'
        starting_cont = c1_b0_cont

        # continuation of eta_p
        #######################

        # step 1: codim 1 investigation
        c1_b1_sols, c1_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params,
                                       name='c1.2:eta_p', NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=starting_cont,
                                       NMX=6000, DSMAX=0.1, bidirectional=True, UZR={2: [2.5]})

        starting_point = 'UZ1'
        starting_cont = c1_b1_cont

        # continuation of eta_a
        #######################

        # step 1: codim 1 investigation
        c1_b2_sols, c1_b2_cont = a.run(starting_point=starting_point, c='qif', ICP=3, NPAR=n_params,
                                       name=f'c1.2:eta_a', NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=starting_cont,
                                       NMX=6000, DSMAX=0.1, bidirectional=True, UZR={3: [0.0]})

        # starting_point = 'UZ1'
        # starting_cont = c1_b2_cont

        # continuation of k_gp
        ######################

        # step 1: codim 1 investigation
        c1_b3_sols, c1_b3_cont = a.run(starting_point=starting_point, c='qif', ICP=22, NPAR=n_params,
                                       name=f'c1.2:k_gp', NDIM=n_dim, RL1=100.0,
                                       origin=starting_cont, NMX=6000, DSMAX=0.1)

        # continuation of k_ap
        ######################

        # step 1: codim 1 investigation
        c1_b4_sols, c1_b4_cont = a.run(starting_point=starting_point, c='qif', ICP=9, NPAR=n_params,
                                       name=f'c1.2:k_ap', NDIM=n_dim, RL0=0.0, RL1=10.0, origin=starting_cont, NMX=6000,
                                       DSMAX=0.1, bidirectional=True)

        # step 2: codim 2 investigation of hopf bifurcation found in step 1
        a.run(starting_point='HB1', c='qif2', ICP=[24, 9], NPAR=n_params, name=f'c1.2:k_i/k_ap',
              NDIM=n_dim, RL0=0.1, RL1=10.0, origin=c1_b4_cont, NMX=4000, DSMAX=0.1, bidirectional=True)
        a.run(starting_point='HB1', c='qif2', ICP=[26, 9], NPAR=n_params, name=f'c1.2:k_gpe_e/k_ap',
              NDIM=n_dim, RL0=0.1, RL1=10.0, origin=c1_b4_cont, NMX=4000, DSMAX=0.1, bidirectional=True)
        a.run(starting_point='HB1', c='qif2', ICP=[8, 9], NPAR=n_params, name=f'c1.2:k_pp/k_ap',
              NDIM=n_dim, RL0=0.1, RL1=10.0, origin=c1_b4_cont, NMX=4000, DSMAX=0.1, bidirectional=True)
        a.run(starting_point='HB1', c='qif2', ICP=[22, 9], NPAR=n_params, name=f'c1.2:k_gp/k_ap',
              NDIM=n_dim, RL0=1.0, RL1=100.0, origin=c1_b4_cont, NMX=4000, DSMAX=0.1, bidirectional=True)

        # step 2: continuation of limit cycle found in step 1
        a.run(starting_point='HB1', c='qif2b', ICP=[9, 11], NPAR=n_params, name=f'c1.2:k_ap_lc',
              NDIM=n_dim, RL1=10.0, origin=c1_b4_cont, NMX=2000, DSMAX=0.2)

        # save results
        kwargs = dict()
        a.to_file(fname, **kwargs)

###############################################################################
# c2: investigation of GPe behavior for balanced inter-population connections #
###############################################################################

if any(c2):

    starting_point = 'UZ3'
    starting_cont = s1_cont

    # step 4: choose balance between STN -> GPe-p vs. STN -> GPe-a projection
    s3_sols, s3_cont = a.run(starting_point=starting_point, c='qif', ICP=26, NPAR=n_params, name='k_gp_e',
                             NDIM=n_dim, RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.2,
                             bidirectional=True, UZR={26: [0.5, 0.75, 1.5, 2.0]}, STOP={})

    #####################################
    # c2.1: STN -> GPe-p < STN -> GPe-a #
    #####################################

    if c2[0]:

        starting_point = 'UZ1'
        starting_cont = s3_cont

        # continuation of eta_e
        #######################

        # step 1: codim 1 investigation
        c2_b0_sols, c2_b0_cont = a.run(starting_point=starting_point, c='qif', ICP=1, NPAR=n_params,
                                       name='c2:eta_e', NDIM=n_dim, RL0=-10.0, RL1=10.0, origin=starting_cont,
                                       NMX=6000, DSMAX=0.1, bidirectional=True, UZR={1: [2.0]})

        starting_point = 'UZ1'
        starting_cont = c2_b0_cont

        # continuation of eta_a
        #######################

        # step 1: codim 1 investigation
        c2_b2_sols, c2_b2_cont = a.run(starting_point=starting_point, c='qif', ICP=3, NPAR=n_params,
                                       name=f'c2:eta_a', NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=starting_cont,
                                       NMX=6000, DSMAX=0.1, bidirectional=True, UZR={3: [-5.0]})

        starting_point = 'UZ1'
        starting_cont = c2_b2_cont

        # continuation of eta_p
        #######################

        # step 1: codim 1 investigation
        c2_b1_sols, c2_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params,
                                       name='c2:eta_p', NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=starting_cont,
                                       NMX=6000, DSMAX=0.1, bidirectional=True, UZR={2: [6.0]})

        starting_point = 'UZ1'
        starting_cont = c2_b1_cont

        # continuation of k_gp
        ######################

        # step 1: codim 1 investigation
        c2_b3_sols, c2_b3_cont = a.run(starting_point=starting_point, c='qif', ICP=22, NPAR=n_params,
                                       name=f'c2:k_gp', NDIM=n_dim, RL1=200.0,
                                       origin=starting_cont, NMX=6000, DSMAX=0.1)

        # continuation of k_p_out
        #########################

        # step 1: codim 1 investigation
        c2_b4_sols, c2_b4_cont = a.run(starting_point=starting_point, c='qif', ICP=25, NPAR=n_params,
                                       name=f'c2:k_p_out', NDIM=n_dim, RL0=0.0, RL1=10.0,
                                       origin=starting_cont, NMX=6000, DSMAX=0.1)

        # continuation of k_p_out
        #########################

        # step 1: codim 1 investigation
        c2_b5_sols, c2_b5_cont = a.run(starting_point=starting_point, c='qif', ICP=29, NPAR=n_params,
                                       name=f'c2:k_a_out', NDIM=n_dim, RL0=0.0, RL1=10.0,
                                       origin=starting_cont, NMX=6000, DSMAX=0.1)

        # save results
        kwargs = dict()
        a.to_file(fname, **kwargs)

    #####################################
    # c2.2: STN -> GPe-p > STN -> GPe-a #
    #####################################

    if c2[1]:

        starting_point = 'UZ4'
        starting_cont = s3_cont

        # continuation of eta_e
        #######################

        # step 1: codim 1 investigation
        c2_b0_sols, c2_b0_cont = a.run(starting_point=starting_point, c='qif', ICP=1, NPAR=n_params,
                                       name='c2.2:eta_e', NDIM=n_dim, RL0=-10.0, RL1=10.0, origin=starting_cont,
                                       NMX=6000, DSMAX=0.1, bidirectional=True, UZR={1: [4.0]})

        starting_point = 'UZ1'
        starting_cont = c2_b0_cont

        # continuation of eta_p
        #######################

        # step 1: codim 1 investigation
        c2_b1_sols, c2_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params,
                                       name='c2.2:eta_p', NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=starting_cont,
                                       NMX=6000, DSMAX=0.1, bidirectional=True, UZR={2: [-3.0]})

        starting_point = 'UZ1'
        starting_cont = c2_b1_cont

        # continuation of eta_a
        #######################

        # step 1: codim 1 investigation
        c2_b2_sols, c2_b2_cont = a.run(starting_point=starting_point, c='qif', ICP=3, NPAR=n_params,
                                       name=f'c2.2:eta_a', NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=starting_cont,
                                       NMX=6000, DSMAX=0.1, bidirectional=True, UZR={3: [0.0]})

        # starting_point = 'UZ1'
        # starting_cont = c1_b2_cont

        # continuation of k_gp
        ######################

        # step 1: codim 1 investigation
        c2_b3_sols, c2_b3_cont = a.run(starting_point=starting_point, c='qif', ICP=22, NPAR=n_params,
                                       name=f'c2.2:k_gp', NDIM=n_dim, RL1=100.0,
                                       origin=starting_cont, NMX=6000, DSMAX=0.1)

        # # step 2: codim 2 investigation of hopf bifurcation found in step 1
        # a.run(starting_point='HB1', c='qif2', ICP=[24, 22], NPAR=n_params, name=f'c2.2:k_gp/k_i',
        #       NDIM=n_dim, RL0=0.1, RL1=10.0, origin=c2_b3_cont, NMX=4000, DSMAX=0.1, bidirectional=True)
        # a.run(starting_point='HB1', c='qif2', ICP=[26, 22], NPAR=n_params, name=f'c2.2:k_gp/k_gp_e',
        #       NDIM=n_dim, RL0=0.1, RL1=10.0, origin=c2_b3_cont, NMX=4000, DSMAX=0.1, bidirectional=True)
        # a.run(starting_point='HB1', c='qif2', ICP=[8, 22], NPAR=n_params, name=f'c2.2:k_gp/k_pp',
        #       NDIM=n_dim, RL0=0.1, RL1=10.0, origin=c2_b3_cont, NMX=4000, DSMAX=0.1, bidirectional=True)
        # a.run(starting_point='HB1', c='qif2', ICP=[9, 22], NPAR=n_params, name=f'c2.2:k_gp/k_ap',
        #       NDIM=n_dim, RL0=1.0, RL1=100.0, origin=c2_b3_cont, NMX=4000, DSMAX=0.1, bidirectional=True)
        # a.run(starting_point='HB1', c='qif2', ICP=[10, 22], NPAR=n_params, name=f'c2.2:k_gp/k_pa',
        #       NDIM=n_dim, RL0=1.0, RL1=100.0, origin=c2_b3_cont, NMX=4000, DSMAX=0.1, bidirectional=True)
        # a.run(starting_point='HB1', c='qif2', ICP=[1, 2], NPAR=n_params, name=f'c2.2:eta_p/eta_e',
        #       NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=c2_b3_cont, NMX=4000, DSMAX=0.1, bidirectional=True)
        #
        # # step 2: continuation of limit cycle found in step 1
        # a.run(starting_point='HB1', c='qif2b', ICP=[22, 11], NPAR=n_params, name=f'c2.2:k_ap_lc',
        #       NDIM=n_dim, RL1=100.0, origin=c2_b3_cont, NMX=2000, DSMAX=0.2)
        # a.run(starting_point='HB2', c='qif2b', ICP=[22, 11], NPAR=n_params, name=f'c2.2:k_ap_lc2',
        #       NDIM=n_dim, RL1=100.0, origin=c2_b3_cont, NMX=2000, DSMAX=0.2)

        # continuation of k_a_out
        #########################

        # step 1: codim 1 investigation
        c2_b4_sols, c2_b4_cont = a.run(starting_point=starting_point, c='qif', ICP=29, NPAR=n_params,
                                       name=f'c2.2:k_a_out', NDIM=n_dim, RL1=10.0,
                                       origin=starting_cont, NMX=6000, DSMAX=0.1)

        # continuation of k_p_out
        #########################

        # step 1: codim 1 investigation
        c2_b5_sols, c2_b5_cont = a.run(starting_point=starting_point, c='qif', ICP=25, NPAR=n_params,
                                       name=f'c2.2:k_p_out', NDIM=n_dim, RL1=10.0,
                                       origin=starting_cont, NMX=6000, DSMAX=0.1)

        # save results
        kwargs = dict()
        a.to_file(fname, **kwargs)

#########################################################################
# c3: investigation of GPe behavior for strong GPe-p <-> GPe-a coupling #
#########################################################################

if any(c3):

    starting_point = 'UZ4'
    starting_cont = s2_cont

    # step 4: choose balance between STN -> GPe-p vs. STN -> GPe-a projection
    s3_sols, s3_cont = a.run(starting_point=starting_point, c='qif', ICP=26, NPAR=n_params, name='k_gp_e',
                             NDIM=n_dim, RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.2,
                             bidirectional=True, UZR={26: [0.5, 0.75, 1.5, 2.0]}, STOP={})

    #####################################
    # c3.1: STN -> GPe-p < STN -> GPe-a #
    #####################################

    if c3[0]:

        starting_point = 'UZ1'
        starting_cont = s3_cont

        # continuation of eta_e
        #######################

        # step 1: codim 1 investigation
        c3_b0_sols, c3_b0_cont = a.run(starting_point=starting_point, c='qif', ICP=1, NPAR=n_params,
                                       name='c3:eta_e', NDIM=n_dim, RL0=-10.0, RL1=10.0, origin=starting_cont,
                                       NMX=6000, DSMAX=0.1, bidirectional=True, UZR={1: [-2.0]})

        starting_point = 'UZ1'
        starting_cont = c3_b0_cont

        # continuation of eta_p
        #######################

        # step 1: codim 1 investigation
        c3_b1_sols, c3_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params,
                                       name='c3:eta_p', NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=starting_cont,
                                       NMX=6000, DSMAX=0.1, bidirectional=True, UZR={2: [4.0]})

        starting_point = 'UZ1'
        starting_cont = c3_b1_cont

        # continuation of eta_a
        #######################

        # step 1: codim 1 investigation
        c3_b2_sols, c3_b2_cont = a.run(starting_point=starting_point, c='qif', ICP=3, NPAR=n_params,
                                       name=f'c3:eta_a', NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=starting_cont,
                                       NMX=6000, DSMAX=0.1, bidirectional=True, UZR={3: [-3.0]})

        starting_point = 'UZ1'
        starting_cont = c3_b2_cont

        # continuation of k_gp
        ######################

        # step 1: codim 1 investigation
        c3_b3_sols, c3_b3_cont = a.run(starting_point=starting_point, c='qif', ICP=22, NPAR=n_params, name=f'c3:k_gp',
                                       NDIM=n_dim, RL1=200.0, origin=starting_cont, NMX=6000, DSMAX=0.1)

        # continuation of k_a_out
        #########################

        # step 1: codim 1 investigation
        c3_b4_sols, c3_b4_cont = a.run(starting_point=starting_point, c='qif', ICP=29, NPAR=n_params, name=f'c3:k_a_out',
                                       NDIM=n_dim, RL0=0.0, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.1)

        # # step 2: codim 2 investigation of hopf bifurcation found in step 1
        # a.run(starting_point='HB1', c='qif2', ICP=[24, 9], NPAR=n_params, name=f'c1:k_i/k_ap',
        #       NDIM=n_dim, RL0=0.1, RL1=10.0, origin=c1_b4_cont, NMX=4000, DSMAX=0.1, bidirectional=True)
        # a.run(starting_point='HB1', c='qif2', ICP=[26, 9], NPAR=n_params, name=f'c1:k_gpe_e/k_ap',
        #       NDIM=n_dim, RL0=0.1, RL1=10.0, origin=c1_b4_cont, NMX=4000, DSMAX=0.1, bidirectional=True)
        # a.run(starting_point='HB1', c='qif2', ICP=[8, 9], NPAR=n_params, name=f'c1:k_pp/k_ap',
        #       NDIM=n_dim, RL0=0.1, RL1=10.0, origin=c1_b4_cont, NMX=4000, DSMAX=0.1, bidirectional=True)
        # a.run(starting_point='HB1', c='qif2', ICP=[22, 9], NPAR=n_params, name=f'c1:k_gp/k_ap',
        #       NDIM=n_dim, RL0=1.0, RL1=100.0, origin=c1_b4_cont, NMX=4000, DSMAX=0.1, bidirectional=True)

        # # step 2: continuation of limit cycle found in step 1
        # a.run(starting_point='HB1', c='qif2b', ICP=[9, 11], NPAR=n_params, name=f'c2:k_ap_lc',
        #       NDIM=n_dim, RL1=10.0, origin=c2_b4_cont, NMX=2000, DSMAX=0.2)

        # continuation of k_p_out
        #########################

        # step 1: codim 1 investigation
        c3_b5_sols, c3_b5_cont = a.run(starting_point=starting_point, c='qif', ICP=25, NPAR=n_params,
                                       name=f'c3:k_p_out',
                                       NDIM=n_dim, RL0=0.0, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.1)

        # save results
        kwargs = dict()
        a.to_file(fname, **kwargs)

    #####################################
    # c3.2: STN -> GPe-p > STN -> GPe-a #
    #####################################

    if c3[1]:

        starting_point = 'UZ4'
        starting_cont = s3_cont

        # continuation of eta_e
        #######################

        # step 1: codim 1 investigation
        c3_b0_sols, c3_b0_cont = a.run(starting_point=starting_point, c='qif', ICP=1, NPAR=n_params,
                                       name='c3.2:eta_e', NDIM=n_dim, RL0=-10.0, RL1=10.0, origin=starting_cont,
                                       NMX=6000, DSMAX=0.1, bidirectional=True, UZR={1: [6.0]})

        starting_point = 'UZ1'
        starting_cont = c3_b0_cont

        # continuation of eta_p
        #######################

        # step 1: codim 1 investigation
        c3_b1_sols, c3_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params,
                                       name='c3.2:eta_p', NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=starting_cont,
                                       NMX=6000, DSMAX=0.1, bidirectional=True, UZR={2: [-5.0]})

        starting_point = 'UZ1'
        starting_cont = c3_b1_cont

        # continuation of eta_a
        #######################

        # step 1: codim 1 investigation
        c3_b2_sols, c3_b2_cont = a.run(starting_point=starting_point, c='qif', ICP=3, NPAR=n_params,
                                       name=f'c3.2:eta_a', NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=starting_cont,
                                       NMX=6000, DSMAX=0.1, bidirectional=True, UZR={3: [4.0]})

        starting_point = 'UZ1'
        starting_cont = c3_b2_cont

        # continuation of k_gp
        ######################

        # step 1: codim 1 investigation
        c3_b3_sols, c3_b3_cont = a.run(starting_point=starting_point, c='qif', ICP=22, NPAR=n_params,
                                       name=f'c3.2:k_gp', NDIM=n_dim, RL1=60.0,
                                       origin=starting_cont, NMX=6000, DSMAX=0.1)

        # 2D bifurcation analysis: k_gp x k_gp_e
        ########################################

        # step 1: codim 2 investigation of hopf bifurcation found in k_gp continuation
        c3_b3_hb2_sols, c3_b3_hb2_cont = a.run(starting_point='HB1', c='qif2', ICP=[26, 22], NPAR=n_params,
                                               name=f'c3.2:k_gp/k_gp_e', NDIM=n_dim, RL0=0.1, RL1=10.0,
                                               origin=c3_b3_cont, NMX=4000, DSMAX=0.1, bidirectional=True)

        # perform branch switching at codimension 2 bifurcations to continue nearby fold/hopf branches
        i, j = 0, 0
        for s in c3_b3_hb2_sols.values():

            if 'ZH' in s['bifurcation']:

                i += 1
                p_start = f"ZH{i}"

                # step 2: perform branch switching
                s_tmp, c_tmp = a.run(starting_point=p_start, c='qif', ICP=26, NPAR=n_params,
                                     name=f'c3.2:k_gp_e/{p_start}', NDIM=n_dim, RL0=0.1, RL1=10.0, NMX=6000, DSMAX=0.1,
                                     origin=c3_b3_hb2_cont, STOP={'LP1'})

                # step 3: continuation of the fold/hopf bifurcations found in step 2 in 2 parameters
                s2_tmp, c2_tmp = a.run(starting_point='LP1', c='qif2', ICP=[26, 22], NPAR=n_params,
                                       name=f'c3.2:k_gp/k_gp_e/{p_start}', NDIM=n_dim, RL0=0.1, RL1=10.0, NMX=6000,
                                       DSMAX=0.1, origin=c_tmp, bidirectional=True)

            if 'GH' in s['bifurcation']:

                j += 1
                p_start = f"GH{j}"

                # step 2: perform branch switching
                s_tmp, c_tmp = a.run(starting_point=p_start, c='qif2b', ICP=[26, 11], NPAR=n_params,
                                     name=f'c3.2:k_gp_e/{p_start}', NDIM=n_dim, RL0=0.1, RL1=10.0, NMX=6000, DSMAX=0.1,
                                     origin=c3_b3_hb2_cont, STOP={'LP1'})

                # step 3: continuation of the fold of limit cycle bifurcation found in step 2 in 2 parameters
                s2_tmp, c2_tmp = a.run(starting_point='LP1', c='qif3', ICP=[26, 22, 11], NPAR=n_params,
                                       name=f'c3.2:k_gp/k_gp_e/{p_start}', NDIM=n_dim, RL0=0.1, RL1=10.0, NMX=2000,
                                       DSMAX=0.5, origin=c_tmp, bidirectional=True)

        # save results
        kwargs = dict()
        a.to_file(fname, **kwargs)

        # 2D bifurcation analysis: k_gp x k_i
        #####################################

        # step 1: codim 2 investigation of hopf bifurcation found in k_gp continuation
        c3_b3_hb1_sols, c3_b3_hb1_cont = a.run(starting_point='HB1', c='qif2', ICP=[24, 22], NPAR=n_params,
                                               name=f'c3.2:k_gp/k_i', NDIM=n_dim, RL0=0.1, RL1=10.0,
                                               origin=c3_b3_cont, NMX=4000, DSMAX=0.1, bidirectional=True)

        # perform branch switching at codimension 2 bifurcations to continue nearby fold/hopf branches
        i, j = 0, 0
        for s in c3_b3_hb1_sols.values():

            if 'ZH' in s['bifurcation']:

                i += 1
                p_start = f"ZH{i}"

                # step 2: perform branch switching
                s_tmp, c_tmp = a.run(starting_point=p_start, c='qif', ICP=24, NPAR=n_params,
                                     name=f'c3.2:k_i/{p_start}', NDIM=n_dim, RL0=0.1, RL1=10.0, NMX=6000, DSMAX=0.1,
                                     origin=c3_b3_hb1_cont, STOP={'LP1'})

                # step 3: continuation of the fold/hopf bifurcations found in step 2 in 2 parameters
                s2_tmp, c2_tmp = a.run(starting_point='LP1', c='qif2', ICP=[24, 22], NPAR=n_params,
                                       name=f'c3.2:k_gp/k_i/{p_start}', NDIM=n_dim, RL0=0.1, RL1=10.0, NMX=6000,
                                       DSMAX=0.1, origin=c_tmp, bidirectional=True)

            if 'GH' in s['bifurcation']:

                j += 1
                p_start = f"GH{j}"

                # step 2: perform branch switching
                s_tmp, c_tmp = a.run(starting_point=p_start, c='qif2b', ICP=[24, 11], NPAR=n_params,
                                     name=f'c3.2:k_i/{p_start}', NDIM=n_dim, RL0=0.1, RL1=10.0, NMX=6000, DSMAX=0.1,
                                     origin=c3_b3_hb1_cont, STOP={'LP1'})

                # step 3: continuation of the fold of limit cycle bifurcation found in step 2 in 2 parameters
                s2_tmp, c2_tmp = a.run(starting_point='LP1', c='qif2', ICP=[24, 22], NPAR=n_params,
                                       name=f'c3.2:k_gp/k_i/{p_start}', NDIM=n_dim, RL0=0.1, RL1=10.0, NMX=6000,
                                       DSMAX=0.1, origin=c_tmp, bidirectional=True)

        # save results
        kwargs = dict()
        a.to_file(fname, **kwargs)

        # 2D bifurcation analysis: k_i x k_gp_e
        #######################################

        # step 1: codim 2 investigation of hopf bifurcation found in k_gp continuation
        c3_b3_hb3_sols, c3_b3_hb3_cont = a.run(starting_point='HB1', c='qif2', ICP=[24, 26], NPAR=n_params,
                                               name=f'c3.2:k_i/k_gp_e', NDIM=n_dim, RL0=0.1, RL1=10.0,
                                               origin=c3_b3_cont, NMX=4000, DSMAX=0.1, bidirectional=True)

        # perform branch switching at codimension 2 bifurcations to continue nearby fold/hopf branches
        i, j = 0, 0
        for s in c3_b3_hb3_sols.values():

            if 'ZH' in s['bifurcation']:

                i += 1
                p_start = f"ZH{i}"

                # step 2: perform branch switching
                s_tmp, c_tmp = a.run(starting_point=p_start, c='qif', ICP=26, NPAR=n_params,
                                     name=f'c3.2:k_gp_e/{p_start}', NDIM=n_dim, RL0=0.1, RL1=10.0, NMX=6000, DSMAX=0.1,
                                     origin=c3_b3_hb3_cont, STOP={'LP1'})

                # step 3: continuation of the fold/hopf bifurcations found in step 2 in 2 parameters
                s2_tmp, c2_tmp = a.run(starting_point='LP1', c='qif2', ICP=[26, 24], NPAR=n_params,
                                       name=f'c3.2:k_i/k_gp_e/{p_start}', NDIM=n_dim, RL0=0.1, RL1=10.0, NMX=6000,
                                       DSMAX=0.1, origin=c_tmp, bidirectional=True)

            if 'GH' in s['bifurcation']:

                j += 1
                p_start = f"GH{j}"

                # step 2: perform branch switching
                s_tmp, c_tmp = a.run(starting_point=p_start, c='qif2b', ICP=[26, 11], NPAR=n_params,
                                     name=f'c3.2:k_gp_e/{p_start}', NDIM=n_dim, RL0=0.1, RL1=10.0, NMX=6000, DSMAX=0.1,
                                     origin=c3_b3_hb3_cont, STOP={'LP1'})

                # step 3: continuation of the fold of limit cycle bifurcation found in step 2 in 2 parameters
                s2_tmp, c2_tmp = a.run(starting_point='LP1', c='qif2', ICP=[26, 24], NPAR=n_params,
                                       name=f'c3.2:k_i/k_gp_e/{p_start}', NDIM=n_dim, RL0=0.1, RL1=10.0, NMX=6000,
                                       DSMAX=0.1, origin=c_tmp, bidirectional=True)

        # step 6: continue curve of hopf bifurcation found in step 5 in 2 parameters
        # a.run(starting_point='HB2', c='qif2', ICP=[26, 22], NPAR=n_params, name='c3.2:k_gp/k_gp_e/zh2', NDIM=n_dim,
        #       RL0=0.1, RL1=10.0, NMX=6000, DSMAX=0.5, origin=c3_b3_zh3_cont, bidirectional=True)
        # a.run(starting_point='HB6', c='qif2', ICP=[26, 22], NPAR=n_params, name='c3.2:k_gp/k_gp_e/zh3', NDIM=n_dim,
        #       RL0=0.1, RL1=10.0, NMX=6000, DSMAX=0.5, origin=c3_b3_zh3_cont, bidirectional=True)
        # a.run(starting_point='HB7', c='qif2', ICP=[26, 22], NPAR=n_params, name='c3.2:k_gp/k_gp_e/zh4', NDIM=n_dim,
        #       RL0=0.1, RL1=10.0, NMX=6000, DSMAX=0.5, origin=c3_b3_zh3_cont, bidirectional=True)
        # a.run(starting_point='HB3', c='qif2', ICP=[26, 22], NPAR=n_params, name='c3.2:k_gp/k_gp_e/zh5', NDIM=n_dim,
        #       RL0=0.1, RL1=10.0, NMX=6000, DSMAX=0.5, origin=c3_b3_zh4_cont, bidirectional=True)
        # a.run(starting_point='HB4', c='qif2', ICP=[26, 22], NPAR=n_params, name='c3.2:k_gp/k_gp_e/zh6', NDIM=n_dim,
        #       RL0=0.1, RL1=10.0, NMX=6000, DSMAX=0.5, origin=c3_b3_zh4_cont, bidirectional=True)

        # # step 8: perform continuation of k_gp for different values of k_gp_e and continue periodic orbits, if found
        # i = 1
        # for point, point_info in c3_b3_zh3_sols.items():
        #     if 'UZ' in point_info['bifurcation']:
        #         sols_tmp, cont_tmp = a.run(starting_point=f'UZ{i}', c='qif', ICP=22, NPAR=n_params, DSMAX=0.1,
        #                                    name=f'c3.2:k_gp_{i}', NDIM=n_dim, RL0=0.0, RL1=300.0, origin=c3_b3_zh3_cont,
        #                                    NMX=6000, bidirectional=True)
        #         sols = [v['bifurcation'] for v in sols_tmp.values()]
        #         n_hopfs = sum([s == 'HB' for s in sols])
        #         for j in range(n_hopfs):
        #             a.run(starting_point=f'HB{j+1}', c='qif2b', ICP=[22, 11], NPAR=n_params, DSMAX=0.2, STOP={'BP2'},
        #                   NDIM=n_dim, name=f'c3.2:k_gp_{i}_lc{j+1}', RL0=0.0, RL1=300.0, origin=cont_tmp, NMX=2000)
        #         i += 1

        # save results
        kwargs = dict()
        a.to_file(fname, **kwargs)
