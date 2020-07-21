from pyrates.utility.pyauto import PyAuto, codim2_search
import numpy as np
import matplotlib.pyplot as plt

"""Bifurcation analysis of GPe model with two populations (arkypallidal and prototypical) and 
gamma-dstributed axonal delays and bi-exponential synapses."""

# config
n_dim = 18
n_params = 23
a = PyAuto("auto_files")

# choice of conditions to run bifurcation analysis for
c1 = False  # strong GPe-p projections
c2 = True  # strong bidirectional coupling between GPe-p and GPe-a
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
                             UZR={19: [10.0, 15.0, 20.0, 25.0]}, STOP={})

    starting_point = 'UZ4'
    starting_cont = s0_cont

    # step 1: codim 1 investigation
    s1_sols, s1_cont = a.run(starting_point=starting_point, c='qif', ICP=20, NPAR=n_params, name='k_p', NDIM=n_dim,
                             RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.1, bidirectional=True,
                             UZR={20: [0.5, 0.75, 1.5, 2.0]}, STOP={})

    starting_point = 'UZ3'
    starting_cont = s1_cont

    # step 1: codim 1 investigation
    k_i_col = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    s2_sols, s2_cont = a.run(starting_point=starting_point, c='qif', ICP=21, NPAR=n_params, name='k_i', NDIM=n_dim,
                             RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.1, bidirectional=True,
                             UZR={21: k_i_col}, STOP={})

    eta_min = -20.0
    eta_max = 20.0
    c = 1
    codim1_col = []
    for i in range(len(k_i_col)):

        if k_i_col[i] == 1.0:
            starting_point = 'UZ3'
            starting_cont = s1_cont
            c = 0
        else:
            starting_point = f'UZ{i+c}'
            starting_cont = s2_cont

        # continuation of eta_p
        #######################

        # step 1: codim 1 investigation
        c1_b1_sols, c1_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params,
                                       name=f'c1:eta_p/v{i}', NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=starting_cont,
                                       NMX=6000, DSMAX=0.1, bidirectional=True, UZR={2: [3.0]})

        starting_point = 'UZ1'
        starting_cont = c1_b1_cont

        # continuation of eta_a
        #######################

        # step 1: codim 1 investigation
        codim1_col.append(a.run(starting_point=starting_point, c='qif', ICP=3, NPAR=n_params,
                                name=f'c1:eta_a/v{i}', NDIM=n_dim, RL0=eta_min, RL1=eta_max,
                                origin=starting_cont, NMX=6000, DSMAX=0.05, bidirectional=True))

    # step 2: codim 2 investigation of hopf and fold bifurcations from step 1
    target_conts = [0, 1, 2, 3, 4, 5, 6]
    kwargs = {'k_i': k_i_col}
    for i, t in enumerate(target_conts):
        sol, cont = codim1_col[t]
        sols = [v['bifurcation'] for v in sol.values()]
        if 'LP' in sols:
            starting_points = ['LP1']
        elif 'HB' in sols:
            starting_points = ['HB1']
        else:
            starting_points = []

        if starting_points:
            c1_b4_sols = codim2_search(params=[7, 3], starting_points=starting_points, origin=cont,
                                       pyauto_instance=a, periodic=False, c='qif', NDIM=n_dim, NPAR=n_params,
                                       RL0=0.1, RL1=10.0, NMX=8000, DSMAX=0.05, max_recursion_depth=3,
                                       name=f"c{i}:k_i/eta_a", kwargs_2D_lc_cont={'c': 'qif3'},
                                       kwargs_2D_cont={'c': 'qif2'}, kwargs_lc_cont={'c': 'qif2b'})
            c1_b5_sols = codim2_search(params=[7, 21], starting_points=starting_points, origin=cont,
                                       pyauto_instance=a, periodic=False, c='qif', NDIM=n_dim, NPAR=n_params, RL0=0.1,
                                       RL1=10.0, NMX=8000, DSMAX=0.05, max_recursion_depth=3, name=f"c{i}:k_i/k_ap",
                                       kwargs_2D_lc_cont={'c': 'qif3'}, kwargs_2D_cont={'c': 'qif2'},
                                       kwargs_lc_cont={'c': 'qif2b'})

            # save results
            kwargs.update({f'c{i}:k_i/eta_a:names': list(c1_b4_sols.keys()),
                           f'c{i}:k_i/k_ap:names': list(c1_b5_sols.keys())})

    # save final results
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
                             UZR={19: [10.0, 20.0, 25.0, 30.0]}, STOP={})

    starting_point = 'UZ3'
    starting_cont = s0_cont

    # step 1: codim 1 investigation
    s1_sols, s1_cont = a.run(starting_point=starting_point, c='qif', ICP=20, NPAR=n_params, name='k_p', NDIM=n_dim,
                             RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.1, bidirectional=True,
                             UZR={20: [0.25, 0.5, 2.0, 4.0]}, STOP={})

    starting_point = 'UZ3'
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

    # continuation of eta_p
    #######################

    # step 1: codim 1 investigation
    c2_b1_sols, c2_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params, name='c2:eta_p',
                                   NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=starting_cont, NMX=6000, DSMAX=0.1,
                                   bidirectional=True, UZR={2: [2.5]})

    # step 2: codim 2 investigation of fold found in step 1
    c2_b1_fp1_sols, c2_b1_fp1_cont = a.run(starting_point='LP1', c='qif2', ICP=[3, 2], NPAR=n_params,
                                           name='c2:eta_p/eta_a', NDIM=n_dim, RL0=-20, RL1=20.0, origin=c2_b1_cont,
                                           NMX=6000, DSMAX=0.5, bidirectional=True)
    c2_b1_fp2_sols, c2_b1_fp2_cont = a.run(starting_point='LP1', c='qif2', ICP=[19, 2], NPAR=n_params,
                                           name='c2:eta_p/k_gp', NDIM=n_dim, RL0=1.0, RL1=100.0, origin=c2_b1_cont,
                                           NMX=6000, DSMAX=0.5, STOP={'CP3'})
    c2_b1_fp3_sols, c2_b1_fp3_cont = a.run(starting_point='LP1', c='qif2', ICP=[20, 2], NPAR=n_params,
                                           name='c2:eta_p/k_p', NDIM=n_dim, RL0=0.1, RL1=10.0, origin=c2_b1_cont,
                                           NMX=6000, DSMAX=0.5, STOP={'CP3'})
    c2_b1_fp4_sols, c2_b1_fp4_cont = a.run(starting_point='LP1', c='qif2', ICP=[21, 2], NPAR=n_params,
                                           name='c2:eta_p/k_i', NDIM=n_dim, RL0=0.1, RL1=10.0, origin=c2_b1_cont,
                                           NMX=6000, DSMAX=0.5, bidirectional=True, STOP={'CP3'})
    c2_b1_fp5_sols, c2_b1_fp5_cont = a.run(starting_point='LP1', c='qif2', ICP=[21, 20], NPAR=n_params,
                                           name='c2:k_p/k_i', NDIM=n_dim, RL0=0.1, RL1=10.0, origin=c2_b1_cont,
                                           NMX=6000, DSMAX=0.5, bidirectional=True, STOP={'CP3'})

    starting_point = 'UZ3'
    starting_cont = c2_b1_cont

    # continuation of eta_a
    #######################

    # step 1: codim 1 investigation
    c2_b2_sols, c2_b2_cont = a.run(starting_point=starting_point, c='qif', ICP=3, NPAR=n_params, name='c2:eta_a',
                                   NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=starting_cont, NMX=6000, DSMAX=0.1,
                                   bidirectional=True)

    # step 2: codim 2 investigation of fold found in step 1
    c2_b2_fp1_sols, c2_b2_fp1_cont = a.run(starting_point='LP1', c='qif2', ICP=[19, 3], NPAR=n_params,
                                           name='c2:eta_a/k_gp', NDIM=n_dim, RL0=1.0, RL1=100.0, origin=c2_b1_cont,
                                           NMX=6000, DSMAX=0.5, STOP={'CP3'}, bidirectional=True)
    c2_b2_fp2_sols, c2_b2_fp2_cont = a.run(starting_point='LP1', c='qif2', ICP=[20, 3], NPAR=n_params,
                                           name='c2:eta_a/k_p', NDIM=n_dim, RL0=0.1, RL1=10.0, origin=c2_b1_cont,
                                           NMX=6000, DSMAX=0.5, STOP={'CP3'}, bidirectional=True)
    c2_b2_fp3_sols, c2_b2_fp3_cont = a.run(starting_point='LP1', c='qif2', ICP=[21, 3], NPAR=n_params,
                                           name='c2:eta_a/k_i', NDIM=n_dim, RL0=0.1, RL1=10.0, origin=c2_b1_cont,
                                           NMX=6000, DSMAX=0.5, bidirectional=True, STOP={'CP3'})
    c2_b2_fp4_sols, c2_b2_fp4_cont = a.run(starting_point='LP1', c='qif2', ICP=[21, 20], NPAR=n_params,
                                           name='c2:k_p/k_i', NDIM=n_dim, RL0=0.1, RL1=10.0, origin=c2_b1_cont,
                                           NMX=6000, DSMAX=0.5, bidirectional=True, STOP={'CP3'})
    c2_b2_fp5_sols, c2_b2_fp5_cont = a.run(starting_point='LP1', c='qif2', ICP=[21, 19], NPAR=n_params,
                                           name='c2:k_gp/k_i', NDIM=n_dim, RL0=0.1, RL1=10.0, origin=c2_b1_cont,
                                           NMX=6000, DSMAX=0.5, bidirectional=True, STOP={'CP3'})

    # continuation of delta_p
    #########################

    # step 1: codim 1 investigation
    c2_b3_sols, c2_b3_cont = a.run(starting_point=starting_point, c='qif', ICP=16, NPAR=n_params, name='c2:delta_p',
                                   NDIM=n_dim, RL0=0.01, RL1=1.0, origin=starting_cont, NMX=6000, DSMAX=0.1,
                                   bidirectional=True)

    # continuation of delta_a
    #########################

    # step 1: codim 1 investigation
    c2_b4_sols, c2_b4_cont = a.run(starting_point=starting_point, c='qif', ICP=17, NPAR=n_params, name='c2:delta_a',
                                   NDIM=n_dim, RL0=0.01, RL1=1.0, origin=starting_cont, NMX=6000, DSMAX=0.1,
                                   bidirectional=True)

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

    starting_point = 'UZ3'
    starting_cont = s0_cont

    # step 1: codim 1 investigation
    s1_sols, s1_cont = a.run(starting_point=starting_point, c='qif', ICP=20, NPAR=n_params, name='k_p', NDIM=n_dim,
                             RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.1, bidirectional=True,
                             UZR={20: [0.25, 0.5, 2.0, 4.0]}, STOP={})

    starting_point = 'UZ3'
    starting_cont = s0_cont

    # step 1: codim 1 investigation
    s2_sols, s2_cont = a.run(starting_point=starting_point, c='qif', ICP=21, NPAR=n_params, name='k_i', NDIM=n_dim,
                             RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.1, bidirectional=True,
                             UZR={21: [0.25, 0.5, 2.0, 4.0]}, STOP={})

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
                                   bidirectional=True, UZR={2: [4.0]})

    starting_point = 'UZ1'
    starting_cont = c3_b2_cont

    # continuation of eta_a
    #######################

    # step 1: codim 1 investigation
    c3_b1_sols, c3_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=3, NPAR=n_params, name='c3:eta_a',
                                   NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=starting_cont, NMX=6000, DSMAX=0.05,
                                   bidirectional=True, UZR={3: [-2.0]})

    # step 2: codim 2 investigation of hopf 1 found in step 1
    c3_b1_hb1_sols, c3_b1_hb1_cont = a.run(starting_point='HB1', c='qif2', ICP=[3, 2], NPAR=n_params,
                                           name='c3:eta_a/eta_p', NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=c3_b1_cont,
                                           NMX=6000, DSMAX=0.1, bidirectional=True)
    c3_b1_hb2_sols, c3_b1_hb2_cont = a.run(starting_point='HB1', c='qif2', ICP=[8, 7], NPAR=n_params,
                                           name='c3:k_pa/k_ap', NDIM=n_dim, RL0=0.0, RL1=10.0, origin=c3_b1_cont,
                                           NMX=6000, DSMAX=0.1, bidirectional=True)
    c3_b1_hb3_sols, c3_b1_hb3_cont = a.run(starting_point='HB1', c='qif2', ICP=[3, 20], NPAR=n_params,
                                           name='c3:eta_a/k_p', NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=c3_b1_cont,
                                           NMX=6000, DSMAX=0.1, bidirectional=True)
    c3_b1_hb4_sols, c3_b1_hb4_cont = a.run(starting_point='HB1', c='qif2', ICP=[3, 21], NPAR=n_params,
                                           name='c3:eta_a/k_i', NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=c3_b1_cont,
                                           NMX=6000, DSMAX=0.1, bidirectional=True)
    c3_b1_hb5_sols, c3_b1_hb5_cont = a.run(starting_point='HB1', c='qif2', ICP=[21, 22], NPAR=n_params,
                                           name='c3:k_i/k_pi', NDIM=n_dim, RL0=0.1, RL1=10.0,
                                           origin=c3_b1_cont, NMX=6000, DSMAX=0.1, bidirectional=True)
    c3_b1_hb6_sols, c3_b1_hb6_cont = a.run(starting_point='HB1', c='qif2', ICP=[21, 20], NPAR=n_params,
                                           name='c3:k_i/k_p', NDIM=n_dim, RL0=0.1, RL1=10.0,
                                           origin=c3_b1_cont, NMX=6000, DSMAX=0.1, bidirectional=True)

    # step 3: continuation of periodic orbit of hopf from step 1
    c3_b1_lc1_sols, c3_b1_lc1_cont = a.run(starting_point='HB1', c='qif2b', ICP=[3, 11], NPAR=n_params,
                                           name='c3:eta_a_lc', NDIM=n_dim, RL0=-20.0, RL1=0.0, origin=c3_b1_cont,
                                           NMX=6000, DSMAX=0.05, STOP={'BP1', 'PD1'})
    c3_b1_lc2_sols, c3_b1_lc2_cont = a.run(starting_point='HB2', c='qif2b', ICP=[3, 11], NPAR=n_params,
                                           name='c3:eta_a_lc2', NDIM=n_dim, RL0=0.0, RL1=20.0, origin=c3_b1_cont,
                                           NMX=6000, DSMAX=0.05, STOP={'BP1', 'PD1', 'LP1'})

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
