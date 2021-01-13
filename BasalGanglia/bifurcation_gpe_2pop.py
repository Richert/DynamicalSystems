from pyrates.utility.pyauto import PyAuto
import sys

"""
Bifurcation analysis of GPe model with two populations (arkypallidal and prototypical) and 
gamma-dstributed axonal delays and bi-exponential synapses. Creates the bifurcation diagrams of Fig. 1 and 2 of
(citation).

To run this code, you need Python >= 3.6 with PyRates (https://github.com/pyrates-neuroscience/PyRates) and 
auto-07p (https://github.com/auto-07p/auto-07p) installed.
"""

path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"

# config
n_dim = 18
n_params = 23
a = PyAuto("auto_files", auto_dir=auto_dir)

# choice of conditions to run bifurcation analysis for
c1 = True  # strong GPe-p projections
c2 = False  # strong bidirectional coupling between GPe-p and GPe-a
c3 = False  # weak bidirectional coupling between GPe-p and GPe-a

################################
# initial continuation in time #
################################

t_sols, t_cont = a.run(e='gpe_2pop', c='ivp', ICP=14, DS=5e-3, DSMIN=1e-4, DSMAX=1.0, NMX=1000000, name='t',
                       UZR={14: 10000.0}, STOP={'UZ1'}, NDIM=n_dim, NPAR=n_params)

##################################################################
# c1: investigation of GPe behavior for strong GPe-p projections #
##################################################################

if c1:

    starting_point = 'UZ1'
    starting_cont = t_cont

    # continuation of GPe intrinsic coupling
    ########################################

    # step 1: codim 1 investigation
    s0_sols, s0_cont = a.run(starting_point=starting_point, c='qif', ICP=19, NPAR=n_params, name='k_gp', NDIM=n_dim,
                             RL0=0.99, RL1=100.0, origin=starting_cont, NMX=6000, DSMAX=0.1,
                             UZR={19: [10.0, 15.0, 20.0, 25.0, 30.0]}, STOP={})

    starting_point = 'UZ5'
    starting_cont = s0_cont

    # step 1: codim 1 investigation
    s1_sols, s1_cont = a.run(starting_point=starting_point, c='qif', ICP=20, NPAR=n_params, name='k_p', NDIM=n_dim,
                             RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.1, bidirectional=True,
                             UZR={20: [0.5, 0.75, 1.5, 2.0]}, STOP={})

    starting_point = 'UZ3'
    starting_cont = s1_cont

    # step 1: codim 1 investigation
    s2_sols, s2_cont = a.run(starting_point=starting_point, c='qif', ICP=21, NPAR=n_params, name='k_i', NDIM=n_dim,
                             RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.1, bidirectional=True,
                             UZR={21: [0.9, 1.8]}, STOP={})

    eta_min = -12.0
    eta_max = 10.0
    c = 1
    kwargs = {}

    # continuation of bifurcation parameters
    ########################################

    # step 1: codim 1 investigation in eta_p
    c1_b1_sols, c1_b1_cont = a.run(starting_point='UZ1', c='qif', ICP=2, NPAR=n_params,
                                   name=f'c1:eta_p/v1', NDIM=n_dim, RL0=0.0, RL1=10.0, origin=s2_cont,
                                   NMX=6000, DSMAX=0.1, UZR={2: [4.8]})
    c1_b2_sols, c1_b2_cont = a.run(starting_point='UZ2', c='qif', ICP=2, NPAR=n_params,
                                   name=f'c1:eta_p/v2', NDIM=n_dim, RL0=0.0, RL1=10.0, origin=s2_cont,
                                   NMX=6000, DSMAX=0.1, UZR={2: [3.2]})

    # step 2: codim 1 investigation in eta_a
    c1_b3_sols, c1_b3_cont = a.run(starting_point='UZ1', c='qif', ICP=3, NPAR=n_params,
                                   name=f'c1:eta_a/v1', NDIM=n_dim, RL0=eta_min, RL1=eta_max,
                                   origin=c1_b1_cont, NMX=8000, DSMAX=0.01, bidirectional=True, UZR={3: [-2.0]})
    c1_b4_sols, c1_b4_cont = a.run(starting_point='UZ1', c='qif', ICP=3, NPAR=n_params,
                                   name=f'c1:eta_a/v2', NDIM=n_dim, RL0=eta_min, RL1=eta_max,
                                   origin=c1_b2_cont, NMX=8000, DSMAX=0.01, bidirectional=True, UZR={3: [3.0]})

    # step 3: codim 1 investigation in k_pa
    c1_b5_sols, c1_b5_cont = a.run(starting_point='UZ1', origin=c1_b3_cont, ICP=8, NPAR=n_params, NDIM=n_dim, RL0=0.0,
                                   RL1=10.0, NMX=8000, DSMAX=0.01, bidirectional=True, name='c1:k_pa/v1')
    c1_b6_sols, c1_b6_cont = a.run(starting_point='UZ1', origin=c1_b4_cont, ICP=8, NPAR=n_params, NDIM=n_dim, RL0=0.0,
                                   RL1=10.0, NMX=8000, DSMAX=0.01, bidirectional=True, name='c1:k_pa/v2')

    # step 3: codim 2 investigation of hopf curves
    c1_b5_2d1_sols, c1_b5_2d1_cont = a.run(starting_point='HB1', origin=c1_b5_cont, c='qif2', ICP=[8, 6], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=8000, DSMAX=0.1,
                                           name='c1:k_pa/k_pp:hb1', bidirectional=True)
    c1_b5_2d2_sols, c1_b5_2d2_cont = a.run(starting_point='HB1', origin=c1_b5_cont, c='qif2', ICP=[8, 7], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=8000, DSMAX=0.1,
                                           name='c1:k_pa/k_ap:hb1', bidirectional=True)

    # step 4: codim 2 investigation of fold curves
    c1_b6_2d1_sols, c1_b6_2d1_cont = a.run(starting_point='LP1', origin=c1_b6_cont, c='qif2', ICP=[8, 6], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=8000, DSMAX=0.1,
                                           name='c1:k_pa/k_pp:lp1', bidirectional=True)
    c1_b6_2d2_sols, c1_b6_2d2_cont = a.run(starting_point='LP1', origin=c1_b6_cont, c='qif2', ICP=[8, 7], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=8000, DSMAX=0.1,
                                           name='c1:k_pa/k_ap:lp1', bidirectional=True)

    c1_b6_2d3_sols, c1_b6_2d3_cont = a.run(starting_point='HB1', origin=c1_b6_cont, c='qif2', ICP=[8, 6], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=8000, DSMAX=0.1,
                                           name='c1:k_pa/k_pp:hb2', bidirectional=True)
    c1_b6_2d4_sols, c1_b6_2d4_cont = a.run(starting_point='HB1', origin=c1_b6_cont, c='qif2', ICP=[8, 7], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=8000, DSMAX=0.1,
                                           name='c1:k_pa/k_ap:hb2', bidirectional=True)

    # step 5: continuation of limit cycles
    c1_b3_lc_sols, c1_b3_lc_cont = a.run(starting_point='HB1', c='qif2b', ICP=[3, 11], NPAR=n_params,
                                         name=f'c1:eta_a/v1:lc', NDIM=n_dim, origin=c1_b3_cont, NMX=2000, DSMAX=0.02,
                                         RL0=-12.0)
    c1_b5_lc_sols, c1_b5_lc_cont = a.run(starting_point='HB1', c='qif2b', ICP=[8, 11], NPAR=n_params,
                                         name=f'c1:k_pa/v1:lc', NDIM=n_dim, origin=c1_b5_cont, NMX=2000, DSMAX=0.02,
                                         RL0=0.0)

    # save results
    fname = '../results/gpe_2pop_c1.pkl'
    a.to_file(fname, **kwargs)

#########################################################################
# c2: investigation of GPe behavior for strong GPe-a <-> GPe-p coupling #
#########################################################################

if c2:

    starting_point = 'UZ1'
    starting_cont = t_cont

    # continuation of GPe intrinsic coupling
    ########################################

    # step 1: codim 1 investigation
    s0_sols, s0_cont = a.run(starting_point=starting_point, c='qif', ICP=19, NPAR=n_params, name='k_gp', NDIM=n_dim,
                             RL0=0.99, RL1=100.0, origin=starting_cont, NMX=6000, DSMAX=0.1,
                             UZR={19: [10.0, 15.0, 20.0, 25.0, 30.0]}, STOP={})

    starting_point = 'UZ5'
    starting_cont = s0_cont

    # step 1: codim 1 investigation
    s1_sols, s1_cont = a.run(starting_point=starting_point, c='qif', ICP=20, NPAR=n_params, name='k_p', NDIM=n_dim,
                             RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.1, bidirectional=True,
                             UZR={20: [0.25, 0.5, 2.0, 4.0]}, STOP={})

    starting_point = 'UZ5'
    starting_cont = s0_cont

    # step 1: codim 1 investigation
    s2_sols, s2_cont = a.run(starting_point=starting_point, c='qif', ICP=21, NPAR=n_params, name='k_i', NDIM=n_dim,
                             RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.1, bidirectional=True,
                             UZR={21: [1.8, 1.9, 2.0, 2.2]}, STOP={})

    starting_point = 'UZ4'
    starting_cont = s2_cont

    # step 1: codim 1 investigation
    s3_sols, s3_cont = a.run(starting_point=starting_point, c='qif', ICP=22, NPAR=n_params, name='k_ap', NDIM=n_dim,
                             RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.1, bidirectional=True,
                             UZR={22: [0.25, 0.5, 2.0, 4.0]}, STOP={})

    starting_point = 'UZ4'
    starting_cont = s2_cont

    # continuation of eta_p
    #######################

    # step 1: codim 1 investigation
    c2_b1_sols, c2_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params, name='c2:eta_p',
                                   NDIM=n_dim, RL0=-30.0, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.05,
                                   bidirectional=True, UZR={2: [4.0]})

    # # step 2: codim 2 investigation of fold found in step 1
    # c2_b1_fp1_sols, c2_b1_fp1_cont = a.run(starting_point='LP1', c='qif2', ICP=[3, 2], NPAR=n_params,
    #                                        name='c2:eta_p/eta_a', NDIM=n_dim, RL0=-20, RL1=20.0, origin=c2_b1_cont,
    #                                        NMX=6000, DSMAX=0.5, bidirectional=True)
    # c2_b1_fp2_sols, c2_b1_fp2_cont = a.run(starting_point='LP1', c='qif2', ICP=[19, 2], NPAR=n_params,
    #                                        name='c2:eta_p/k_gp', NDIM=n_dim, RL0=1.0, RL1=100.0, origin=c2_b1_cont,
    #                                        NMX=6000, DSMAX=0.5, STOP={'CP3'})
    # c2_b1_fp3_sols, c2_b1_fp3_cont = a.run(starting_point='LP1', c='qif2', ICP=[20, 2], NPAR=n_params,
    #                                        name='c2:eta_p/k_p', NDIM=n_dim, RL0=0.1, RL1=10.0, origin=c2_b1_cont,
    #                                        NMX=6000, DSMAX=0.5, STOP={'CP3'})
    # c2_b1_fp4_sols, c2_b1_fp4_cont = a.run(starting_point='LP1', c='qif2', ICP=[21, 2], NPAR=n_params,
    #                                        name='c2:eta_p/k_i', NDIM=n_dim, RL0=0.1, RL1=10.0, origin=c2_b1_cont,
    #                                        NMX=6000, DSMAX=0.5, bidirectional=True, STOP={'CP3'})
    # c2_b1_fp5_sols, c2_b1_fp5_cont = a.run(starting_point='LP1', c='qif2', ICP=[21, 20], NPAR=n_params,
    #                                        name='c2:k_p/k_i', NDIM=n_dim, RL0=0.1, RL1=10.0, origin=c2_b1_cont,
    #                                        NMX=6000, DSMAX=0.5, bidirectional=True, STOP={'CP3'})

    starting_point = 'UZ1'
    starting_cont = c2_b1_cont

    # continuation of eta_a
    #######################

    # step 1: codim 1 investigation
    c2_b2_sols, c2_b2_cont = a.run(starting_point=starting_point, c='qif', ICP=3, NPAR=n_params, name='c2:eta_a',
                                   NDIM=n_dim, RL0=-40.0, RL1=20.0, origin=starting_cont, NMX=8000, DSMAX=0.01,
                                   bidirectional=True)

    # # step 2: codim 2 investigation of fold found in step 1
    # c2_b2_fp1_sols, c2_b2_fp1_cont = a.run(starting_point='LP1', c='qif2', ICP=[8, 3], NPAR=n_params,
    #                                        name='c2:eta_a/k_pa', NDIM=n_dim, RL0=0.0, RL1=10.0, origin=c2_b2_cont,
    #                                        NMX=6000, DSMAX=0.1, STOP={'CP3'}, bidirectional=True)
    # c2_b2_fp2_sols, c2_b2_fp2_cont = a.run(starting_point='LP1', c='qif2', ICP=[6, 3], NPAR=n_params,
    #                                        name='c2:eta_a/k_pp', NDIM=n_dim, RL0=0.0, RL1=10.0, origin=c2_b2_cont,
    #                                        NMX=6000, DSMAX=0.1, STOP={'CP3'}, bidirectional=True)
    # c2_b2_fp3_sols, c2_b2_fp3_cont = a.run(starting_point='LP1', c='qif2', ICP=[8, 6], NPAR=n_params,
    #                                        name='c2:k_pp/k_pa', NDIM=n_dim, RL0=0.0, RL1=10.0, origin=c2_b2_cont,
    #                                        NMX=6000, DSMAX=0.1, bidirectional=True, STOP={'CP3'})

    # save results
    fname = '../results/gpe_2pop_c2.pkl'
    kwargs = dict()
    a.to_file(fname, **kwargs)

#######################################################################
# c3: investigation of GPe behavior for weak GPe-p <-> GPe-a coupling #
#######################################################################

if c3:

    starting_point = 'UZ1'
    starting_cont = t_cont

    # continuation of GPe intrinsic coupling
    ########################################

    # step 1: codim 1 investigation
    s0_sols, s0_cont = a.run(starting_point=starting_point, c='qif', ICP=19, NPAR=n_params, name='k_gp', NDIM=n_dim,
                             RL0=0.99, RL1=100.0, origin=starting_cont, NMX=6000, DSMAX=0.1,
                             UZR={19: [10.0, 20.0, 25.0, 30.0]}, STOP={})

    starting_point = 'UZ4'
    starting_cont = s0_cont

    # step 1: codim 1 investigation
    s1_sols, s1_cont = a.run(starting_point=starting_point, c='qif', ICP=20, NPAR=n_params, name='k_p', NDIM=n_dim,
                             RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.1, bidirectional=True,
                             UZR={20: [0.25, 0.5, 2.0, 4.0]}, STOP={})

    starting_point = 'UZ4'
    starting_cont = s0_cont

    # step 1: codim 1 investigation
    s2_sols, s2_cont = a.run(starting_point=starting_point, c='qif', ICP=21, NPAR=n_params, name='k_i', NDIM=n_dim,
                             RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.1, bidirectional=True,
                             UZR={21: [0.5, 0.7, 2.0, 4.0]}, STOP={})

    starting_point = 'UZ2'
    starting_cont = s2_cont

    # step 1: codim 1 investigation
    s3_sols, s3_cont = a.run(starting_point=starting_point, c='qif', ICP=22, NPAR=n_params, name='k_ap', NDIM=n_dim,
                             RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.1, bidirectional=True,
                             UZR={22: [0.25, 0.5, 2.0, 4.0]}, STOP={})

    starting_point = 'UZ2'
    starting_cont = s2_cont

    # continuation of eta_p
    #######################

    # step 1: codim 1 investigation
    c3_b2_sols, c3_b2_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params, name='c3:eta_p',
                                   NDIM=n_dim, RL0=-5.0, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.1,
                                   bidirectional=True, UZR={2: [4.5]})

    starting_point = 'UZ1'
    starting_cont = c3_b2_cont

    # continuation of eta_a
    #######################

    # step 1: codim 1 investigation
    c3_b1_sols, c3_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=3, NPAR=n_params, name='c3:eta_a',
                                   NDIM=n_dim, RL0=-25.0, RL1=15.0, origin=starting_cont, NMX=8000, DSMAX=0.01,
                                   bidirectional=True, UZR={})

    # step 2: codim 2 investigation of hopf 1 found in step 1
    # c3_b1_hb1_sols, c3_b1_hb1_cont = a.run(starting_point='HB1', c='qif2', ICP=[8, 3], NPAR=n_params,
    #                                        name='c3:eta_a/k_pa', NDIM=n_dim, RL0=0.0, RL1=10.0, origin=c3_b1_cont,
    #                                        NMX=6000, DSMAX=0.1, bidirectional=True)
    # c3_b1_hb2_sols, c3_b1_hb2_cont = a.run(starting_point='HB1', c='qif2', ICP=[6, 3], NPAR=n_params,
    #                                        name='c3:eta_a/k_pp', NDIM=n_dim, RL0=0.0, RL1=10.0, origin=c3_b1_cont,
    #                                        NMX=6000, DSMAX=0.1, bidirectional=True)
    # c3_b1_hb3_sols, c3_b1_hb3_cont = a.run(starting_point='HB1', c='qif2', ICP=[8, 6], NPAR=n_params,
    #                                        name='c3:k_pp/k_pa', NDIM=n_dim, RL0=0.0, RL1=10.0, origin=c3_b1_cont,
    #                                        NMX=6000, DSMAX=0.1, bidirectional=True)

    # step 3: continuation of periodic orbit of hopf from step 1
    c3_b1_lc1_sols, c3_b1_lc1_cont = a.run(starting_point='HB1', c='qif2b', ICP=[3, 11], NPAR=n_params,
                                           name='c3:eta_a_lc', NDIM=n_dim, RL0=-12.0, RL1=0.0, origin=c3_b1_cont,
                                           NMX=6000, DSMAX=0.02, STOP={'BP1', 'PD1'})
    c3_b1_lc2_sols, c3_b1_lc2_cont = a.run(starting_point='HB2', c='qif2b', ICP=[3, 11], NPAR=n_params,
                                           name='c3:eta_a_lc2', NDIM=n_dim, RL0=0.0, RL1=12.0, origin=c3_b1_cont,
                                           NMX=6000, DSMAX=0.02, STOP={'BP1', 'PD1', 'LP3'})

    starting_point = 'UZ1'
    starting_cont = c3_b1_cont

    # continuation of delta_a
    #########################

    # # step 1: codim 1 investigation
    # c3_b4_sols, c3_b4_cont = a.run(starting_point=starting_point, c='qif', ICP=17, NPAR=n_params, name='c3:delta_a',
    #                                NDIM=n_dim, RL0=0.01, RL1=1.0, origin=starting_cont, NMX=6000, DSMAX=0.1,
    #                                bidirectional=True)
    #
    # # step 2: codim 2 investigation of hopf found in step 1
    # c3_b4_hb1_sols, c3_b4_hb1_cont = a.run(starting_point='HB2', c='qif2', ICP=[17, 20], NPAR=n_params,
    #                                        name='c3:delta_a/k_p', NDIM=n_dim, RL0=1e-4, RL1=1.0, origin=c3_b4_cont,
    #                                        NMX=6000, DSMAX=0.5, bidirectional=True)
    # c3_b4_hb2_sols, c3_b4_hb2_cont = a.run(starting_point='HB2', c='qif2', ICP=[17, 21], NPAR=n_params,
    #                                        name='c3:delta_a/k_i', NDIM=n_dim, RL0=1e-4, RL1=1.0, origin=c3_b4_cont,
    #                                        NMX=6000, DSMAX=0.5, bidirectional=True)
    # c3_b4_hb3_sols, c3_b4_hb3_cont = a.run(starting_point='HB2', c='qif2', ICP=[7, 8], NPAR=n_params,
    #                                        name='c3:k_pa/k_ap/delta_a', NDIM=n_dim, RL0=0.0, RL1=10.0,
    #                                        origin=c3_b4_cont, NMX=6000, DSMAX=0.5, bidirectional=True)
    # c3_b4_hb4_sols, c3_b4_hb4_cont = a.run(starting_point='HB2', c='qif2', ICP=[21, 22], NPAR=n_params,
    #                                        name='c3:k_i/k_pi/delta_a', NDIM=n_dim, RL0=0.1, RL1=10.0,
    #                                        origin=c3_b4_cont, NMX=6000, DSMAX=0.5, bidirectional=True)
    #
    # # step 3: continuation of periodic orbit of hopf from step 1
    # c3_b3_lc1_sols, c3_b3_lc1_cont = a.run(starting_point='HB1', c='qif2b', ICP=[17, 11], NPAR=n_params,
    #                                        name='c3:delta_a_lc', NDIM=n_dim, RL0=1e-4, RL1=1.0, origin=c3_b4_cont,
    #                                        NMX=6000, DSMAX=0.1, DSMIN=1e-4, STOP={'BP1', 'PD1'})
    # c3_b3_lc2_sols, c3_b3_lc2_cont = a.run(starting_point='HB2', c='qif2b', ICP=[17, 11], NPAR=n_params,
    #                                        name='c3:delta_a_lc2', NDIM=n_dim, RL0=1e-4, RL1=1.0, origin=c3_b4_cont,
    #                                        NMX=6000, DSMAX=0.1, DSMIN=1e-4, STOP={'BP1', 'PD1'})

    # save results
    fname = '../results/gpe_2pop_c3.pkl'
    kwargs = dict()
    a.to_file(fname, **kwargs)
