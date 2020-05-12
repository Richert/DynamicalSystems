from pyauto import PyAuto
import numpy as np
import matplotlib.pyplot as plt

"""Bifurcation analysis of STN-GPe model with two GPe populations (arkypallidal and prototypical) and 
gamma-dstributed axonal delays and bi-exponential synapses."""

# config
n_dim = 37
n_params = 30
a = PyAuto("auto_files")
fname = '../results/stn_gpe_syns.pkl'

# choice of conditions to run bifurcation analysis for
c1 = [  # strong GPe-p projections
      False,  # STN -> GPe-p > STN -> GPe-a
      False,   # STN -> GPe-p == STN -> GPe-a
]
c2 = [  # strong bidirectional coupling between GPe-p and GPe-a
      False,  # STN -> GPe-p > STN -> GPe-a
      False,   # STN -> GPe-p == STN -> GPe-a
]
c3 = [  # strong GPe-p to GPe-a projection
      True,  # STN -> GPe-p > STN -> GPe-a
      False,   # STN -> GPe-p == STN -> GPe-a
]

################################
# initial continuation in time #
################################

t_sols, t_cont = a.run(e='stn_gpe_syns', c='ivp', ICP=14, DS=5e-3, DSMIN=1e-4, DSMAX=1.0, NMX=1000000, name='t',
                       UZR={14: 10000.0}, STOP={'UZ1'}, NDIM=n_dim, NPAR=n_params)

##################################################################
# c1: investigation of GPe behavior for strong GPe-p projections #
##################################################################

if any(c1):

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
                             UZR={23: [0.25, 0.5, 2.0, 4.0]}, STOP={})

    starting_point = 'UZ3'
    starting_cont = s1_cont

    # step 3: choose relative strength of inter- vs. intra-population coupling inside GPe
    s2_sols, s2_cont = a.run(starting_point=starting_point, c='qif', ICP=24, NPAR=n_params, name='k_i', NDIM=n_dim,
                             RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.2, bidirectional=True,
                             UZR={24: [0.25, 0.5, 2.0, 4.0]}, STOP={})

    starting_point = 'UZ3'
    starting_cont = s2_cont

    # step 4: choose relative strength of GPe-p -> GPe-a vs. GPe-a -> GPe-p projection
    s3_sols, s3_cont = a.run(starting_point=starting_point, c='qif', ICP=25, NPAR=n_params, name='k_pi', NDIM=n_dim,
                             RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.2, bidirectional=True,
                             UZR={25: [0.25, 0.5, 2.0, 4.0]}, STOP={})

    starting_point = 'UZ3'
    starting_cont = s2_cont

    # step 5: choose balance between STN -> GPe-p vs. STN -> GPe-a projection
    s4_sols, s4_cont = a.run(starting_point=starting_point, c='qif', ICP=26, NPAR=n_params, name='k_gp_e', NDIM=n_dim,
                             RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.2, bidirectional=True,
                             UZR={26: [0.25, 0.5, 2.0, 4.0]}, STOP={})

    #####################################
    # c1.1: STN -> GPe-p > STN -> GPe-a #
    #####################################

    if c1[0]:

        starting_point = 'UZ3'
        starting_cont = s4_cont

        # continuation of eta_p
        #######################

        # step 1: codim 1 investigation
        c1_b2_sols, c1_b2_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params, name='c1:eta_p',
                                       NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=starting_cont, NMX=6000, DSMAX=0.1,
                                       bidirectional=True, UZR={2: [-4.0]})

        starting_point = 'UZ1'
        starting_cont = c1_b2_cont

        # continuation of eta_a
        #######################

        # step 1: codim 1 investigation
        c1_b3_sols, c1_b3_cont = a.run(starting_point=starting_point, c='qif', ICP=3, NPAR=n_params, name='c1:eta_a',
                                       NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=starting_cont, NMX=6000, DSMAX=0.1,
                                       bidirectional=True, UZR={3: [6.5]})

        # step 2: codim 2 investigation of fold found in step 1
        c1_b3_fp1_sols, c1_b3_fp1_cont = a.run(starting_point='LP1', c='qif2', ICP=[2, 3], NPAR=n_params,
                                               name='c1:eta_a/eta_p', NDIM=n_dim, RL0=-10, RL1=20.0, origin=c1_b3_cont,
                                               NMX=6000, DSMAX=0.5, bidirectional=True)
        c1_b3_fp2_sols, c1_b3_fp2_cont = a.run(starting_point='LP1', c='qif2', ICP=[1, 3], NPAR=n_params,
                                               name='c1:eta_a/eta_e', NDIM=n_dim, RL0=-10.0, RL1=20.0,
                                               origin=c1_b3_cont, NMX=6000, DSMAX=0.5, bidirectional=True, STOP={'ZH3'})

        # step 3: continuation of the codim 2 bifurcations found in step 2
        c1_b3_zh1_sols, c1_b3_zh1_cont = a.run(starting_point='ZH1', c='qif2a', ICP=[2, 3], NPAR=n_params,
                                               name='c1:eta_a/eta_p/zh1', NDIM=n_dim, RL0=-10.0, RL1=20.0,
                                               origin=c1_b3_fp1_cont, NMX=1000, DSMAX=0.5, STOP={'BP1'})
        c1_b3_zh2_sols, c1_b3_zh2_cont = a.run(starting_point='ZH2', c='qif2a', ICP=[2, 3], NPAR=n_params,
                                               name='c1:eta_a/eta_p/zh2', NDIM=n_dim, RL0=-10.0, RL1=20.0,
                                               origin=c1_b3_fp1_cont, NMX=1000, DSMAX=0.5, STOP={'BP1'})
        c1_b3_bt1_sols, c1_b3_bt1_cont = a.run(starting_point='BT1', c='qif2a', ICP=[1, 3], NPAR=n_params,
                                               name='c1:eta_a/eta_e/bt1', NDIM=n_dim, RL0=-10.0, RL1=20.0,
                                               origin=c1_b3_fp2_cont, NMX=1000, DSMAX=0.5, STOP={'BP1'})
        c1_b3_bt2_sols, c1_b3_bt2_cont = a.run(starting_point='BT2', c='qif2a', ICP=[1, 3], NPAR=n_params,
                                               name='c1:eta_a/eta_e/bt2', NDIM=n_dim, RL0=-10.0, RL1=20.0,
                                               origin=c1_b3_fp2_cont, NMX=1000, DSMAX=0.5, STOP={'BP1'})

        starting_point = 'UZ1'
        starting_cont = c1_b3_cont

        # continuation in eta_e
        #######################

        # step 1: codim 1 investigation of STN -> GPe
        c1_b1_sols, c1_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=1, NPAR=n_params, name='c1:eta_e',
                                       NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=starting_cont, NMX=6000, DSMAX=0.05,
                                       bidirectional=True)

        # step 2: codim 2 investigation of hopf bifurcation found in step 1
        c1_b1_hb1_sols, c1_b1_hb1_cont = a.run(starting_point='HB1', c='qif2', ICP=[2, 1], NPAR=n_params,
                                               name='c1:eta_e/eta_p', NDIM=n_dim, RL0=-10.0, RL1=20.0,
                                               origin=c1_b1_cont, NMX=6000, DSMAX=0.5, bidirectional=True)
        c1_b1_hb2_sols, c1_b1_hb2_cont = a.run(starting_point='HB1', c='qif2', ICP=[3, 1], NPAR=n_params,
                                               name='c1:eta_e/eta_a', NDIM=n_dim, RL0=-10.0, RL1=20.0,
                                               origin=c1_b1_cont, NMX=6000, DSMAX=0.5, bidirectional=True)

        # step 3: continuation of the codim 2 bifurcations found in step 2
        c1_b1_gh1_sols, c1_b1_gh1_cont = a.run(starting_point='GH1', c='qif2a', ICP=[2, 1], NPAR=n_params,
                                               name='c1:eta_e/eta_p/gh1', NDIM=n_dim, RL0=-10.0, RL1=20.0,
                                               origin=c1_b1_hb1_cont, NMX=1000, DSMAX=0.5, STOP={'LP3'})
        c1_b1_gh2_sols, c1_b1_gh2_cont = a.run(starting_point='GH2', c='qif2a', ICP=[2, 1], NPAR=n_params,
                                               name='c1:eta_e/eta_p/gh2', NDIM=n_dim, RL0=-10.0, RL1=20.0,
                                               origin=c1_b1_hb1_cont, NMX=1000, DSMAX=0.5, STOP={'LP3'})
        c1_b1_zh1_sols, c1_b1_zh1_cont = a.run(starting_point='ZH1', c='qif2a', ICP=[3, 1], NPAR=n_params,
                                               name='c1:eta_e/eta_a/zh1', NDIM=n_dim, RL0=-10.0, RL1=20.0,
                                               origin=c1_b1_hb2_cont, NMX=1000, DSMAX=0.5, STOP={'BP1'})
        c1_b1_bt1_sols, c1_b1_bt1_cont = a.run(starting_point='BT1', c='qif2a', ICP=[3, 1], NPAR=n_params,
                                               name='c1:eta_e/eta_a/bt1', NDIM=n_dim, RL0=-10.0, RL1=20.0,
                                               origin=c1_b1_hb2_cont, NMX=1000, DSMAX=0.5, STOP={'BP1'}, ILP=0)

        # step 4: continuation of periodic orbit of hopf bifurcation found in step 1
        c1_b1_lc1_sols, c1_b1_lc1_cont = a.run(starting_point='HB1', c='qif2b', ICP=[1, 11], NPAR=n_params,
                                               name='c1:eta_e_lc', NDIM=n_dim, RL0=-10.0, RL1=15.0, origin=c1_b1_cont,
                                               NMX=6000, DSMAX=0.5, STOP={'BP1', 'PD1'})

        # continuation of delta_p
        #########################

        # step 1: codim 1 investigation
        c1_b4_sols, c1_b4_cont = a.run(starting_point=starting_point, c='qif', ICP=19, NPAR=n_params, name='c1:delta_p',
                                       NDIM=n_dim, RL0=0.01, RL1=1.0, origin=starting_cont, NMX=6000, DSMAX=0.1,
                                       bidirectional=True)

        # continuation of delta_a
        #########################

        # step 1: codim 1 investigation
        c1_b5_sols, c1_b5_cont = a.run(starting_point=starting_point, c='qif', ICP=20, NPAR=n_params, name='c1:delta_a',
                                       NDIM=n_dim, RL0=0.01, RL1=1.0, origin=starting_cont, NMX=6000, DSMAX=0.1,
                                       bidirectional=True)

        # save results
        kwargs = dict()
        a.to_file(fname, **kwargs)

    #####################################
    # c1.2: STN -> GPe-p == STN -> GPe-a #
    #####################################

    if c1[1]:

        starting_point = 'UZ3'
        starting_cont = s2_cont

        # continuation of eta_a
        #######################

        # step 1: codim 1 investigation
        c12_b3_sols, c12_b3_cont = a.run(starting_point=starting_point, c='qif', ICP=3, NPAR=n_params,
                                         name='c1.2:eta_a', NDIM=n_dim, RL0=-15.0, RL1=15.0, origin=starting_cont,
                                         NMX=6000, DSMAX=0.1, bidirectional=True, UZR={3: [4.0]})

        starting_point = 'UZ1'
        starting_cont = c12_b3_cont

        # continuation of eta_p
        #######################

        # step 1: codim 1 investigation
        c12_b2_sols, c12_b2_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params,
                                         name='c1.2:eta_p', NDIM=n_dim, RL0=-15.0, RL1=15.0, origin=starting_cont,
                                         NMX=6000, DSMAX=0.1, bidirectional=True, UZR={2: [1.6]})

        starting_point = 'UZ3'
        starting_cont = c12_b2_cont

        # continuation in eta_e
        #######################

        # step 1: codim 1 investigation of STN -> GPe
        c12_b1_sols, c12_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=1, NPAR=n_params,
                                         name='c1.2:eta_e', NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=starting_cont,
                                         NMX=6000, DSMAX=0.1, bidirectional=True)

        # step 2: codim 2 investigation of hopf bifurcation found in step 1
        c12_b1_hb1_sols, c12_b1_hb1_cont = a.run(starting_point='HB1', c='qif2', ICP=[2, 1], NPAR=n_params,
                                                 name='c1.2:eta_e/eta_p/hb1', NDIM=n_dim, RL0=-20.0, RL1=20.0,
                                                 origin=c12_b1_cont, NMX=6000, DSMAX=0.5, bidirectional=True)
        c12_b1_hb2_sols, c12_b1_hb2_cont = a.run(starting_point='HB1', c='qif2', ICP=[3, 1], NPAR=n_params,
                                                 name='c1.2:eta_e/eta_a/hb1', NDIM=n_dim, RL0=-20.0, RL1=20.0,
                                                 origin=c12_b1_cont, NMX=6000, DSMAX=0.5, bidirectional=True)
        c12_b1_hb3_sols, c12_b1_hb3_cont = a.run(starting_point='HB5', c='qif2', ICP=[2, 1], NPAR=n_params,
                                                 name='c1.2:eta_e/eta_p/hb5', NDIM=n_dim, RL0=-20.0, RL1=20.0,
                                                 origin=c12_b1_cont, NMX=6000, DSMAX=0.5, bidirectional=True)
        c12_b1_hb4_sols, c12_b1_hb4_cont = a.run(starting_point='HB5', c='qif2', ICP=[3, 1], NPAR=n_params,
                                                 name='c1.2:eta_e/eta_a/hb5', NDIM=n_dim, RL0=-20.0, RL1=20.0,
                                                 origin=c12_b1_cont, NMX=6000, DSMAX=0.5, bidirectional=True)

        # step 3: codim 2 investigation of fold bifurcation found in step 1
        c12_b1_lp1_sols, c12_b1_lp1_cont = a.run(starting_point='LP1', c='qif2', ICP=[2, 1], NPAR=n_params,
                                                 name='c1.2:eta_e/eta_p/lp1', NDIM=n_dim, RL0=-20.0, RL1=20.0,
                                                 origin=c12_b1_cont, NMX=6000, DSMAX=0.5, bidirectional=True)
        c12_b1_lp2_sols, c12_b1_lp2_cont = a.run(starting_point='LP1', c='qif2', ICP=[3, 1], NPAR=n_params,
                                                 name='c1.2:eta_e/eta_a/lp1', NDIM=n_dim, RL0=-20.0, RL1=20.0,
                                                 origin=c12_b1_cont, NMX=6000, DSMAX=0.5, bidirectional=True)

        # step 4: continuation of periodic orbit of hopf bifurcation found in step 1
        c12_b1_lc1_sols, c12_b1_lc1_cont = a.run(starting_point='HB1', c='qif2b', ICP=[1, 11], NPAR=n_params,
                                                 name='c1.2:eta_e_lc1', NDIM=n_dim, RL0=-20.0, RL1=20.0,
                                                 origin=c12_b1_cont, NMX=6000, DSMAX=0.5, STOP={'BP1', 'PD1', 'TR2'})
        c12_b1_lc2_sols, c12_b1_lc2_cont = a.run(starting_point='HB5', c='qif2b', ICP=[1, 11], NPAR=n_params,
                                                 name='c1.2:eta_e_lc2', NDIM=n_dim, RL0=-20.0, RL1=20.0,
                                                 origin=c12_b1_cont, NMX=6000, DSMAX=0.5, STOP={'BP1', 'PD1', 'TR2'})

        # continuation of delta_p
        #########################

        # step 1: codim 1 investigation
        c12_b4_sols, c12_b4_cont = a.run(starting_point=starting_point, c='qif', ICP=19, NPAR=n_params,
                                         name='c1.2:delta_p', NDIM=n_dim, RL0=0.01, RL1=1.0, origin=starting_cont,
                                         NMX=6000, DSMAX=0.1, bidirectional=True)

        # continuation of delta_a
        #########################

        # step 1: codim 1 investigation
        c12_b5_sols, c12_b5_cont = a.run(starting_point=starting_point, c='qif', ICP=20, NPAR=n_params,
                                         name='c1.2:delta_a', NDIM=n_dim, RL0=0.01, RL1=1.0, origin=starting_cont,
                                         NMX=6000, DSMAX=0.1, bidirectional=True)

        # save results
        kwargs = dict()
        a.to_file(fname, **kwargs)

#########################################################################
# c2: investigation of GPe behavior for strong GPe-a <-> GPe-p coupling #
#########################################################################

if any(c2):

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
                             UZR={23: [0.25, 0.5, 2.0, 4.0]}, STOP={})

    starting_point = 'UZ2'
    starting_cont = s0_cont

    # step 3: choose relative strength of inter- vs. intra-population coupling inside GPe
    s2_sols, s2_cont = a.run(starting_point=starting_point, c='qif', ICP=24, NPAR=n_params, name='k_i', NDIM=n_dim,
                             RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.2, bidirectional=True,
                             UZR={24: [0.25, 0.5, 2.0, 4.0]}, STOP={})

    starting_point = 'UZ3'
    starting_cont = s2_cont

    # step 4: choose relative strength of GPe-p -> GPe-a vs. GPe-a -> GPe-p projection
    s3_sols, s3_cont = a.run(starting_point=starting_point, c='qif', ICP=25, NPAR=n_params, name='k_pi', NDIM=n_dim,
                             RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.2, bidirectional=True,
                             UZR={25: [0.25, 0.5, 2.0, 4.0]}, STOP={})

    starting_point = 'UZ3'
    starting_cont = s2_cont

    # step 5: choose balance between STN -> GPe-p vs. STN -> GPe-a projection
    s4_sols, s4_cont = a.run(starting_point=starting_point, c='qif', ICP=26, NPAR=n_params, name='k_gp_e', NDIM=n_dim,
                             RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.2, bidirectional=True,
                             UZR={26: [0.25, 0.5, 2.0, 4.0]}, STOP={})

    starting_point = 'UZ3'
    starting_cont = s4_cont

    # continuation of eta_p
    #######################

    # step 1: codim 1 investigation
    c2_b2_sols, c2_b2_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params, name='c2:eta_p',
                                   NDIM=n_dim, RL0=-15.0, RL1=15.0, origin=starting_cont, NMX=6000, DSMAX=0.1,
                                   bidirectional=True, UZR={2: [-0.84]})

    # step 2: codim 2 investigation of fold found in step 1
    c2_b2_fp1_sols, c2_b2_fp1_cont = a.run(starting_point='LP1', c='qif2', ICP=[1, 2], NPAR=n_params,
                                           name='c1:eta_p/eta_e', NDIM=n_dim, RL0=-10, RL1=20.0, origin=c2_b2_cont,
                                           NMX=6000, DSMAX=0.5, bidirectional=True)
    c2_b2_fp2_sols, c2_b2_fp2_cont = a.run(starting_point='LP1', c='qif2', ICP=[3, 2], NPAR=n_params,
                                           name='c1:eta_p/eta_a', NDIM=n_dim, RL0=-10.0, RL1=20.0, origin=c2_b2_cont,
                                           NMX=6000, DSMAX=0.5, bidirectional=True)

    starting_point = 'UZ1'
    starting_cont = c2_b2_cont

    # continuation in eta_e
    #######################

    # step 1: codim 1 in eta_e
    c2_b1_sols, c2_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=1, NPAR=n_params, name='c2:eta_e',
                                   NDIM=n_dim, RL0=-10.0, RL1=15.0, origin=starting_cont, NMX=6000, DSMAX=0.1,
                                   bidirectional=True)

    # continuation of eta_a
    #######################

    # step 1: codim 1 investigation
    c2_b3_sols, c2_b3_cont = a.run(starting_point=starting_point, c='qif', ICP=3, NPAR=n_params, name='c2:eta_a',
                                   NDIM=n_dim, RL0=-15.0, RL1=15.0, origin=starting_cont, NMX=6000, DSMAX=0.1,
                                   bidirectional=True)

    # continuation of delta_p
    #########################

    # step 1: codim 1 investigation
    c2_b4_sols, c2_b4_cont = a.run(starting_point=starting_point, c='qif', ICP=19, NPAR=n_params, name='c2:delta_p',
                                   NDIM=n_dim, RL0=0.01, RL1=1.0, origin=starting_cont, NMX=6000, DSMAX=0.05,
                                   bidirectional=True)

    # continuation of delta_a
    #########################

    # step 1: codim 1 investigation
    c2_b5_sols, c2_b5_cont = a.run(starting_point=starting_point, c='qif', ICP=20, NPAR=n_params, name='c2:delta_a',
                                   NDIM=n_dim, RL0=0.01, RL1=1.0, origin=starting_cont, NMX=6000, DSMAX=0.05,
                                   bidirectional=True)

    # save results
    kwargs = dict()
    a.to_file(fname, **kwargs)

########################################################################
# c3: investigation of GPe behavior for strong GPe-p to GPe-a coupling #
########################################################################

if any(c3):

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
                             UZR={23: [0.25, 0.5, 2.0, 4.0]}, STOP={})

    starting_point = 'UZ2'
    starting_cont = s0_cont

    # step 3: choose relative strength of inter- vs. intra-population coupling inside GPe
    s2_sols, s2_cont = a.run(starting_point=starting_point, c='qif', ICP=24, NPAR=n_params, name='k_i', NDIM=n_dim,
                             RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.2, bidirectional=True,
                             UZR={24: [0.25, 0.5, 2.0, 4.0]}, STOP={})

    starting_point = 'UZ2'
    starting_cont = s2_cont

    # step 4: choose relative strength of GPe-p -> GPe-a vs. GPe-a -> GPe-p projection
    s3_sols, s3_cont = a.run(starting_point=starting_point, c='qif', ICP=25, NPAR=n_params, name='k_pi', NDIM=n_dim,
                             RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.2, bidirectional=True,
                             UZR={25: [0.25, 0.5, 2.0, 4.0]}, STOP={})

    starting_point = 'UZ3'
    starting_cont = s3_cont

    # step 5: choose balance between STN -> GPe-p vs. STN -> GPe-a projection
    s4_sols, s4_cont = a.run(starting_point=starting_point, c='qif', ICP=26, NPAR=n_params, name='k_gp_e', NDIM=n_dim,
                             RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.2, bidirectional=True,
                             UZR={26: [0.25, 0.5, 2.0, 4.0]}, STOP={})

    #####################################
    # c1.1: STN -> GPe-p > STN -> GPe-a #
    #####################################

    if c3[0]:

        starting_point = 'UZ3'
        starting_cont = s4_cont

        # continuation of eta_p
        #######################

        # step 1: codim 1 investigation
        c3_b2_sols, c3_b2_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params, name='c3:eta_p',
                                       NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=starting_cont, NMX=6000, DSMAX=0.1,
                                       bidirectional=True, UZR={2: [-1.0]})

        starting_point = 'UZ1'
        starting_cont = c3_b2_cont

        # continuation of eta_a
        #######################

        # step 1: codim 1 investigation
        c3_b3_sols, c3_b3_cont = a.run(starting_point=starting_point, c='qif', ICP=3, NPAR=n_params, name='c3:eta_a',
                                       NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=starting_cont, NMX=6000, DSMAX=0.1,
                                       bidirectional=True)

        # continuation in eta_e
        #######################

        # step 1: codim 1 investigation of STN -> GPe
        c3_b1_sols, c3_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=1, NPAR=n_params, name='c3:eta_e',
                                       NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=starting_cont, NMX=6000, DSMAX=0.05,
                                       bidirectional=True)

        # continuation of delta_p
        #########################

        # step 1: codim 1 investigation
        c3_b4_sols, c3_b4_cont = a.run(starting_point=starting_point, c='qif', ICP=19, NPAR=n_params, name='c3:delta_p',
                                       NDIM=n_dim, RL0=0.01, RL1=1.0, origin=starting_cont, NMX=6000, DSMAX=0.1,
                                       bidirectional=True)

        # continuation of delta_a
        #########################

        # step 1: codim 1 investigation
        c3_b5_sols, c3_b5_cont = a.run(starting_point=starting_point, c='qif', ICP=20, NPAR=n_params, name='c3:delta_a',
                                       NDIM=n_dim, RL0=0.01, RL1=1.0, origin=starting_cont, NMX=6000, DSMAX=0.1,
                                       bidirectional=True)

        # save results
        kwargs = dict()
        a.to_file(fname, **kwargs)
