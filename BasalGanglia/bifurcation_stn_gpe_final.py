from pyrates.utility.pyauto import PyAuto, codim2_search
import numpy as np
import matplotlib.pyplot as plt

"""Bifurcation analysis of STN-GPe model with two GPe populations (arkypallidal and prototypical) and 
gamma-dstributed axonal delays and bi-exponential synapses."""

# config
n_dim = 37
n_params = 29
a = PyAuto("auto_files")

# choice of conditions to run bifurcation analysis for
c1 = [  # bistable state
      False,  # STN -> GPe-p < STN -> GPe-a
      False,   # STN -> GPe-p > STN -> GPe-a
      #False   # STN -> GPe-p == STN -> GPe-a
]
c2 = [  # oscillatory state
    True,  # STN -> GPe-p < STN -> GPe-a
    False,  # STN -> GPe-p > STN -> GPe-a
]

#########################
# initial continuations #
#########################

# continuation in time
######################

t_sols, t_cont = a.run(e='stn_gpe_final', c='ivp', ICP=14, NMX=1000000, name='t', UZR={14: 1000.0}, STOP={'UZ1'},
                       NDIM=n_dim, NPAR=n_params)

starting_point = 'UZ1'
starting_cont = t_cont

# continuation of STN excitability
##################################

e0_sols, e0_cont = a.run(starting_point=starting_point, c='qif', ICP=1, NPAR=n_params, name='eta_e', NDIM=n_dim,
                         RL0=-10.0, RL1=0.1, origin=starting_cont, NMX=4000, DSMAX=0.5, DS='-', UZR={1: [-4.0]})

starting_point = 'UZ1'
starting_cont = e0_cont

# continuation of GPe intrinsic coupling
########################################

# step 1: choose base level of GPe coupling strength
s0_sols, s0_cont = a.run(starting_point=starting_point, c='qif', ICP=22, NPAR=n_params, name='k_gp', NDIM=n_dim,
                         RL0=0.99, RL1=5.0, origin=starting_cont, NMX=2000, DSMAX=0.1,
                         UZR={22: [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]}, STOP={})

starting_point = 'UZ4'
starting_cont = s0_cont

# step 2: choose relative projection strength of GPe-p vs. GPe-a
s1_sols, s1_cont = a.run(starting_point=starting_point, c='qif', ICP=23, NPAR=n_params, name='k_p', NDIM=n_dim,
                         RL0=0.4, RL1=3.0, origin=starting_cont, NMX=2000, DSMAX=0.1, bidirectional=True,
                         UZR={23: [0.5, 0.75, 1.5, 2.0]}, STOP={})

starting_point = 'UZ3'
starting_cont = s1_cont

# step 3: choose relative strength of inter- vs. intra-population coupling inside GPe
s2_sols, s2_cont = a.run(starting_point=starting_point, c='qif', ICP=24, NPAR=n_params, name='k_i', NDIM=n_dim,
                         RL0=0.8, RL1=2.0, origin=starting_cont, NMX=2000, DSMAX=0.1, bidirectional=True,
                         UZR={24: [0.9, 1.8]}, STOP={})

##############################################################################
# c1: investigation of STN-GPe behavior near the bi-stable regime of the GPe #
##############################################################################

if any(c1):

    starting_point = 'UZ2'
    starting_cont = s2_cont

    # choose balance between STN -> GPe-p vs. STN -> GPe-a projection
    #################################################################

    s3_sols, s3_cont = a.run(starting_point=starting_point, c='qif', ICP=26, NPAR=n_params, name='c1:k_gp_e',
                             NDIM=n_dim, RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.2,
                             bidirectional=True, UZR={26: [0.5, 0.75, 1.5, 2.0]}, STOP={})

    if c1[0]:

        fname = '../results/stn_gpe_final_c11.pkl'
        starting_point = 'UZ2'
        starting_cont = s3_cont

        # continuation of eta_p
        #######################

        c1_b0_sols, c1_b0_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params,
                                       name='c1.1:eta_p', NDIM=n_dim, RL0=-10.0, RL1=10.0, origin=starting_cont,
                                       NMX=4000, DSMAX=0.1, bidirectional=True, UZR={2: [5.3]})

        starting_point = 'UZ1'
        starting_cont = c1_b0_cont

        # continuation of eta_a
        #######################

        c1_b1_sols, c1_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=3, NPAR=n_params,
                                       name='c1.1:eta_a', NDIM=n_dim, RL0=-10.0, RL1=20.0, origin=starting_cont,
                                       NMX=4000, DSMAX=0.1, bidirectional=True, UZR={3: [2.6]})

        starting_point = 'UZ1'
        starting_cont = c1_b1_cont

        # investigation of effects of feedback from GPe-p to STN
        ########################################################

        # step 1: increase background excitability of STN
        c1_b2_sols, c1_b2_cont = a.run(starting_point=starting_point, c='qif', ICP=1, NPAR=n_params,
                                       name='c1.1:eta_e', NDIM=n_dim, RL0=-5.0, RL1=3.5, origin=starting_cont,
                                       NMX=8000, DSMAX=0.4, bidirectional=True, UZR={1: [3.0]})

        starting_point = 'UZ1'
        starting_cont = c1_b2_cont

        # step 2: increase coupling strength from GPe-p to STN
        c1_b3_sols, c1_b3_cont = a.run(starting_point=starting_point, c='qif', ICP=7, NPAR=n_params,
                                       name='c1.1:k_ep', NDIM=n_dim, RL0=0.0, RL1=40.0, origin=starting_cont,
                                       NMX=8000, DSMAX=0.4, bidirectional=True, UZR={7: [9.0]})

        starting_point = 'UZ3'
        starting_cont = c1_b3_cont

        # investigation of effects of GPe inhibition
        ############################################

        # step 1: k_ps continuation
        c1_b4_sols, c1_b4_cont = a.run(starting_point=starting_point, c='qif', ICP=16, NPAR=n_params,
                                       name='c1.2:k_ps', NDIM=n_dim, RL0=0.1, RL1=200.0, origin=starting_cont,
                                       NMX=4000, DSMAX=0.1)

        # step 2: k_as continuation
        c1_b5_sols, c1_b5_cont = a.run(starting_point=starting_point, c='qif', ICP=17, NPAR=n_params,
                                       name='c1.2:k_as', NDIM=n_dim, RL0=0.1, RL1=200.0, origin=starting_cont,
                                       NMX=4000, DSMAX=0.1)

        # step 3: k_gp continuation
        c1_b6_sols, c1_b6_cont = a.run(starting_point=starting_point, c='qif', ICP=22, NPAR=n_params,
                                       name='c1.2:k_gp', NDIM=n_dim, RL0=0.0, RL1=20.0, origin=starting_cont,
                                       NMX=4000, DSMAX=0.1)

        # save results
        kwargs = dict()
        a.to_file(fname, **kwargs)

    if c1[1]:

        fname = '../results/stn_gpe_final_c12.pkl'
        starting_point = 'UZ3'
        starting_cont = s3_cont

        # continuation of eta_p
        #######################

        c1_b0_sols, c1_b0_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params,
                                       name='c1.2:eta_p', NDIM=n_dim, RL0=-10.0, RL1=10.0, origin=starting_cont,
                                       NMX=4000, DSMAX=0.1, bidirectional=True, UZR={2: [2.5]})

        starting_point = 'UZ1'
        starting_cont = c1_b0_cont

        # continuation of eta_a
        #######################

        c1_b1_sols, c1_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=3, NPAR=n_params,
                                       name='c1.2:eta_a', NDIM=n_dim, RL0=-10.0, RL1=20.0, origin=starting_cont,
                                       NMX=4000, DSMAX=0.1, bidirectional=True, UZR={3: [4.5]})

        starting_point = 'UZ1'
        starting_cont = c1_b1_cont

        # investigation of effects of feedback from GPe-p to STN
        ########################################################

        # step 1: increase background excitability of STN
        c1_b2_sols, c1_b2_cont = a.run(starting_point=starting_point, c='qif', ICP=1, NPAR=n_params,
                                       name='c1.2:eta_e', NDIM=n_dim, RL0=-5.0, RL1=3.5, origin=starting_cont,
                                       NMX=8000, DSMAX=0.4, bidirectional=True, UZR={1: [3.0]})

        starting_point = 'UZ1'
        starting_cont = c1_b2_cont

        # step 2: increase coupling strength from GPe-p to STN
        c1_b3_sols, c1_b3_cont = a.run(starting_point=starting_point, c='qif', ICP=7, NPAR=n_params,
                                       name='c1.2:k_ep', NDIM=n_dim, RL0=0.0, RL1=30.0, origin=starting_cont,
                                       NMX=8000, DSMAX=0.4, bidirectional=True, UZR={7: [9.0]})

        starting_point = 'UZ1'
        starting_cont = c1_b3_cont

        # investigation of effects of GPe inhibition
        ############################################

        # step 1: k_ps continuation
        c1_b4_sols, c1_b4_cont = a.run(starting_point=starting_point, c='qif', ICP=16, NPAR=n_params,
                                       name='c1.2:k_ps', NDIM=n_dim, RL0=19.0, RL1=200.0, origin=starting_cont,
                                       NMX=4000, DSMAX=0.1)

        # step 2: k_as continuation
        c1_b5_sols, c1_b5_cont = a.run(starting_point=starting_point, c='qif', ICP=17, NPAR=n_params,
                                       name='c1.2:k_as', NDIM=n_dim, RL0=19.0, RL1=200.0, origin=starting_cont,
                                       NMX=4000, DSMAX=0.1)

        # step 3: k_gp continuation
        c1_b6_sols, c1_b6_cont = a.run(starting_point=starting_point, c='qif', ICP=22, NPAR=n_params,
                                       name='c1.2:k_gp', NDIM=n_dim, RL0=0.0, RL1=20.0, origin=starting_cont,
                                       NMX=4000, DSMAX=0.1)

        # bifurcation analysis of hopf bifurcation identified in previous continuation
        ##############################################################################

        # step 1: 2D continuation in k_gp_e and k_gp
        c1_b6_cd2_1 = codim2_search(params=[26, 22], starting_points=['HB1'], origin=c1_b6_cont, pyauto_instance=a,
                                    periodic=False, c='qif', NDIM=n_dim, NPAR=n_params, RL0=0.1, RL1=2.0, NMX=8000,
                                    DSMAX=0.06, max_recursion_depth=2, name="c1.2:k_gp/k_gp_e",
                                    kwargs_2D_lc_cont={'c': 'qif3'}, kwargs_2D_cont={'c': 'qif2'},
                                    kwargs_lc_cont={'c': 'qif2b'}, STOP={'CP3', 'GH5', 'ZH5'})

        kwargs = {'k_gp/k_gp_e:names': list(c1_b6_cd2_1.keys())}
        a.to_file(fname, **kwargs)

        # step 2: continuation of limit cycle
        c1_b6_lc_sols, c1_b6_lc_cont = a.run(starting_point='HB1', c='qif2b', ICP=[22, 11], NPAR=n_params,
                                             name='c1.2:k_gp/lc', NDIM=n_dim, RL0=0.0, RL1=20.0, origin=c1_b6_cont,
                                             NMX=2000, DSMAX=0.2)

        a.to_file(fname, **kwargs)


################################################################################
# c2: investigation of STN-GPe behavior near the oscillatory regime of the GPe #
################################################################################

if any(c2):

    starting_point = 'UZ1'
    starting_cont = s2_cont

    # choose balance between STN -> GPe-p vs. STN -> GPe-a projection
    #################################################################

    s3_sols, s3_cont = a.run(starting_point=starting_point, c='qif', ICP=26, NPAR=n_params, name='c2:k_gp_e',
                             NDIM=n_dim, RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.2,
                             bidirectional=True, UZR={26: [0.5, 0.75, 1.5, 2.0]}, STOP={})

    if c2[0]:

        fname = '../results/stn_gpe_final_c21.pkl'
        starting_point = 'UZ2'
        starting_cont = s3_cont

        # continuation of eta_p
        #######################

        c2_b0_sols, c2_b0_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params,
                                       name='c2.1:eta_p', NDIM=n_dim, RL0=-10.0, RL1=10.0, origin=starting_cont,
                                       NMX=4000, DSMAX=0.1, bidirectional=True, UZR={2: [7.0]})

        starting_point = 'UZ1'
        starting_cont = c2_b0_cont

        # continuation of eta_a
        #######################

        c2_b1_sols, c2_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=3, NPAR=n_params,
                                       name='c2.1:eta_a', NDIM=n_dim, RL0=-10.0, RL1=20.0, origin=starting_cont,
                                       NMX=4000, DSMAX=0.1, bidirectional=True, UZR={3: [-3.0]})

        starting_point = 'UZ1'
        starting_cont = c2_b1_cont

        # investigation of effects of feedback from GPe-p to STN
        ########################################################

        # step 1: increase background excitability of STN
        c2_b2_sols, c2_b2_cont = a.run(starting_point=starting_point, c='qif', ICP=1, NPAR=n_params,
                                       name='c2.1:eta_e', NDIM=n_dim, RL0=-5.0, RL1=3.5, origin=starting_cont,
                                       NMX=8000, DSMAX=0.4, bidirectional=True, UZR={1: [3.0]})

        starting_point = 'UZ1'
        starting_cont = c2_b2_cont

        # step 2: increase coupling strength from GPe-p to STN
        c2_b3_sols, c2_b3_cont = a.run(starting_point=starting_point, c='qif', ICP=7, NPAR=n_params,
                                       name='c2.1:k_ep', NDIM=n_dim, RL0=0.0, RL1=40.0, origin=starting_cont,
                                       NMX=8000, DSMAX=0.4, bidirectional=True, UZR={7: [9.0]})

        starting_point = 'UZ1'
        starting_cont = c2_b3_cont

        # investigation of effects of GPe inhibition
        ############################################

        # step 1: k_ps continuation
        c2_b4_sols, c2_b4_cont = a.run(starting_point=starting_point, c='qif', ICP=16, NPAR=n_params,
                                       name='c2.2:k_ps', NDIM=n_dim, RL0=0.1, RL1=200.0, origin=starting_cont,
                                       NMX=4000, DSMAX=0.1)

        # step 2: k_as continuation
        c2_b5_sols, c2_b5_cont = a.run(starting_point=starting_point, c='qif', ICP=17, NPAR=n_params,
                                       name='c2.2:k_as', NDIM=n_dim, RL0=0.1, RL1=200.0, origin=starting_cont,
                                       NMX=4000, DSMAX=0.1)

        # step 3: k_gp continuation
        c2_b6_sols, c2_b6_cont = a.run(starting_point=starting_point, c='qif', ICP=22, NPAR=n_params,
                                       name='c2.2:k_gp', NDIM=n_dim, RL0=0.0, RL1=20.0, origin=starting_cont,
                                       NMX=4000, DSMAX=0.1)

        # save results
        kwargs = dict()
        a.to_file(fname, **kwargs)

    if c2[1]:

        fname = '../results/stn_gpe_final_c22.pkl'
        starting_point = 'UZ3'
        starting_cont = s3_cont

        # continuation of eta_p
        #######################

        c2_b0_sols, c2_b0_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params,
                                       name='c2.2:eta_p', NDIM=n_dim, RL0=-10.0, RL1=10.0, origin=starting_cont,
                                       NMX=4000, DSMAX=0.1, bidirectional=True, UZR={2: [4.2]})

        starting_point = 'UZ1'
        starting_cont = c2_b0_cont

        # continuation of eta_a
        #######################

        c2_b1_sols, c2_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=3, NPAR=n_params,
                                       name='c2.2:eta_a', NDIM=n_dim, RL0=-10.0, RL1=20.0, origin=starting_cont,
                                       NMX=4000, DSMAX=0.1, bidirectional=True, UZR={3: [-4.0]})

        starting_point = 'UZ1'
        starting_cont = c2_b1_cont

        # investigation of effects of feedback from GPe-p to STN
        ########################################################

        # step 1: increase background excitability of STN
        c2_b2_sols, c2_b2_cont = a.run(starting_point=starting_point, c='qif', ICP=1, NPAR=n_params,
                                       name='c2.2:eta_e', NDIM=n_dim, RL0=-5.0, RL1=4.0, origin=starting_cont,
                                       NMX=8000, DSMAX=0.4, bidirectional=True, UZR={1: [3.5]})

        starting_point = 'UZ1'
        starting_cont = c2_b2_cont

        # step 2: increase coupling strength from GPe-p to STN
        c2_b3_sols, c2_b3_cont = a.run(starting_point=starting_point, c='qif', ICP=7, NPAR=n_params,
                                       name='c2.2:k_ep', NDIM=n_dim, RL0=0.0, RL1=30.0, origin=starting_cont,
                                       NMX=8000, DSMAX=0.4, bidirectional=True, UZR={7: [8.5]})

        starting_point = 'UZ1'
        starting_cont = c2_b3_cont

        # investigation of effects of GPe inhibition
        ############################################

        # step 1: k_ps continuation
        c2_b4_sols, c2_b4_cont = a.run(starting_point=starting_point, c='qif', ICP=16, NPAR=n_params,
                                       name='c2.2:k_ps', NDIM=n_dim, RL0=19.0, RL1=200.0, origin=starting_cont,
                                       NMX=4000, DSMAX=0.1)

        # step 2: k_as continuation
        c2_b5_sols, c2_b5_cont = a.run(starting_point=starting_point, c='qif', ICP=17, NPAR=n_params,
                                       name='c2.2:k_as', NDIM=n_dim, RL0=19.0, RL1=200.0, origin=starting_cont,
                                       NMX=4000, DSMAX=0.1)

        # step 3: k_gp continuation
        c2_b6_sols, c2_b6_cont = a.run(starting_point=starting_point, c='qif', ICP=22, NPAR=n_params,
                                       name='c2.2:k_gp', NDIM=n_dim, RL0=0.0, RL1=20.0, origin=starting_cont,
                                       NMX=4000, DSMAX=0.1)

        # step 4: eta_a continuation
        c2_b7_sols, c2_b7_cont = a.run(starting_point=starting_point, c='qif', ICP=3, NPAR=n_params,
                                       name='c2.2:k_gp', NDIM=n_dim, RL0=-20.0, RL1=0.0, origin=starting_cont,
                                       NMX=4000, DSMAX=0.1, DS='-')

        # bifurcation analysis of hopf bifurcation identified in previous continuation
        ##############################################################################

        # step 1: 2D continuation in k_gp_e and k_gp
        c2_b6_cd2_1 = codim2_search(params=[26, 22], starting_points=['HB1'], origin=c2_b6_cont, pyauto_instance=a,
                                    periodic=False, c='qif', NDIM=n_dim, NPAR=n_params, RL0=0.1, RL1=2.0, NMX=8000,
                                    DSMAX=0.06, max_recursion_depth=2, name="c2.2:k_gp/k_gp_e",
                                    kwargs_2D_lc_cont={'c': 'qif3'}, kwargs_2D_cont={'c': 'qif2'},
                                    kwargs_lc_cont={'c': 'qif2b'}, STOP={'CP3', 'GH5', 'ZH5'})

        kwargs = {'k_gp/k_gp_e:names': list(c2_b6_cd2_1.keys())}
        a.to_file(fname, **kwargs)

        # step 2: continuation of limit cycle
        c2_b6_lc_sols, c2_b6_lc_cont = a.run(starting_point='HB1', c='qif2b', ICP=[22, 11], NPAR=n_params,
                                             name='c2.2:k_gp/lc', NDIM=n_dim, RL0=0.0, RL1=15.0, origin=c2_b6_cont,
                                             NMX=2000, DSMAX=0.2)

        a.to_file(fname, **kwargs)
