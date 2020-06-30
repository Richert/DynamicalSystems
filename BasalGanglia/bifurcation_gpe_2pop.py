from pyauto import PyAuto
import numpy as np
import matplotlib.pyplot as plt

"""Bifurcation analysis of GPe model with two populations (arkypallidal and prototypical) and 
gamma-dstributed axonal delays and bi-exponential synapses."""

# config
n_dim = 18
n_params = 23
a = PyAuto("auto_files")

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
                             UZR={19: [10.0, 15.0, 20.0, 25.0]}, STOP={})

    starting_point = 'UZ3'
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

    c = 1
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
        c1_b2_sols, c1_b2_cont = a.run(starting_point=starting_point, c='qif', ICP=3, NPAR=n_params,
                                       name=f'c1:eta_a/v{i}', NDIM=n_dim, RL0=-12.0, RL1=12.0, origin=starting_cont,
                                       NMX=6000, DSMAX=0.05, bidirectional=True)

        # step 2: codim 2 investigation of hopf and fold bifurcations from step 1
        sols = [v['bifurcation'] for v in c1_b2_sols.values()]
        if 'LP' in sols:
            # a.run(starting_point='LP1', c='qif2', ICP=[21, 7], NPAR=n_params, name=f'c1:k_i/k_ap/LP1/v{i}', NDIM=n_dim,
            #       RL0=0.1, RL1=10.0, origin=c1_b2_cont, NMX=6000, DSMAX=0.1, bidirectional=True)
            a.run(starting_point='LP1', c='qif2', ICP=[3, 21], NPAR=n_params, name=f'c1:eta_a/k_i/LP1/v{i}',
                  NDIM=n_dim, RL0=-12.0, RL1=12.0, origin=c1_b2_cont, NMX=8000, DSMAX=0.1, bidirectional=True)
            # a.run(starting_point='LP1', c='qif2', ICP=[21, 6], NPAR=n_params, name=f'c1:k_i/k_pp/LP1/v{i}', NDIM=n_dim,
            #       RL0=0.1, RL1=10.0, origin=c1_b2_cont, NMX=6000, DSMAX=0.1, bidirectional=True)
            sol_tmp1, cont_tmp1 = a.run(starting_point='LP1', c='qif2', ICP=[3, 7], NPAR=n_params,
                                        name=f'c1:eta_a/k_ap/LP1/v{i}', NDIM=n_dim, RL0=-12.0, RL1=12.0,
                                        origin=c1_b2_cont, NMX=8000, DSMAX=0.1, bidirectional=True)
            sols_tmp = [v['bifurcation'] for v in sol_tmp1.values()]

            if 'ZH' in sols_tmp:
                c1_b2_zh1_sols, c1_b2_zh1_cont = a.run(starting_point='ZH1', c='qif', ICP=3, NPAR=n_params,
                                                       name=f'c1:eta_a/zh1/v{i}', NDIM=n_dim, RL0=-12.0, RL1=12.0,
                                                       NMX=2000, DSMAX=0.5, origin=cont_tmp1, bidirectional=True)
                c1_b2_zh2_sols, c1_b2_zh2_cont = a.run(starting_point='HB1', c='qif2', ICP=[3, 7], NPAR=n_params,
                                                       name=f'c1:eta_a/k_ap/zh1/v{i}', NDIM=n_dim, RL0=-12.0, RL1=12.0,
                                                       NMX=6000, DSMAX=0.1, origin=c1_b2_zh1_cont, bidirectional=True)

        if 'HB' in sols:
            a.run(starting_point='HB1', c='qif2', ICP=[3, 21], NPAR=n_params, name=f'c1:eta_a/k_i/HB1/v{i}',
                  NDIM=n_dim, RL0=-12.0, RL1=12.0, origin=c1_b2_cont, NMX=8000, DSMAX=0.1, bidirectional=True)
            sol_tmp2, cont_tmp2 = a.run(starting_point='HB1', c='qif2', ICP=[3, 7], NPAR=n_params,
                                        name=f'c1:eta_a/k_ap/HB1/v{i}', NDIM=n_dim, RL0=-12.0, RL1=12.0,
                                        origin=c1_b2_cont, NMX=8000, DSMAX=0.1, bidirectional=True)
            sols_tmp = [v['bifurcation'] for v in sol_tmp2.values()]
            if 'ZH' in sols_tmp:
                c1_b2_zh3_sols, c1_b2_zh3_cont = a.run(starting_point='ZH1', c='qif', ICP=3, NPAR=n_params,
                                                       name=f'c1:eta_a/zh2/v{i}', NDIM=n_dim, RL0=-12.0, RL1=12.0,
                                                       NMX=2000, DSMAX=0.5, origin=cont_tmp2, bidirectional=True)
                c1_b2_zh4_sols, c1_b2_zh4_cont = a.run(starting_point='LP1', c='qif2', ICP=[3, 7], NPAR=n_params,
                                                       name=f'c1:eta_a/k_ap/zh2/v{i}', NDIM=n_dim, RL0=-12.0, RL1=12.0,
                                                       NMX=6000, DSMAX=0.1, origin=c1_b2_zh3_cont, bidirectional=True)
                if 'GH' in sols_tmp:
                    n_ghs = sum([s == 'GH' for s in sols_tmp])
                    for j in range(n_ghs):
                        c1_b2_gh1_sols, c1_b2_gh1_cont = a.run(starting_point=f'GH{j+1}', c='qif2b', ICP=[7, 11],
                                                               NPAR=n_params, name=f'c1:k_ap/v{i}/gh{j+1}', NDIM=n_dim,
                                                               RL0=0.0, RL1=10.0, NMX=2000, DSMAX=0.05,
                                                               origin=cont_tmp2, STOP={'LP1'})
                        try:
                            s_tmp, c_tmp = a.run(starting_point='LP1', c='qif3', ICP=[7, 3, 11], NPAR=n_params, RL0=0.0,
                                                 RL1=10.0, NDIM=n_dim,  NMX=1000, origin=c1_b2_gh1_cont, DSMAX=0.1,
                                                 name=f'c1:eta_a/k_ap/v{i}/gh{j + 1}', STOP={'BP9'})
                        except (KeyError, ValueError):
                            pass

        # continuation in k_ap
        ######################

        # # step 1: continuation in k_ap
        # c1_b5_sols, c1_b5_cont = a.run(starting_point=starting_point, c='qif', ICP=7, NPAR=n_params,
        #                                name=f'c1:k_ap/v{i}', NDIM=n_dim, RL0=0.1, RL1=10.0, origin=starting_cont,
        #                                NMX=6000, DSMAX=0.1, bidirectional=True)
        #
        # # step 2: continuation of limit cycle from hopf bifurcation found in step 1
        # if 'HB' in [v['bifurcation'] for v in c1_b5_sols.values()]:
        #     c1_b5_sols, c1_b5_cont = a.run(starting_point='HB1', c='qif2b', ICP=[7, 11], NPAR=n_params,
        #                                    name=f'c1:k_ap_lc/v{i}', NDIM=n_dim, RL0=0.1, RL1=10.0, origin=c1_b5_cont,
        #                                    NMX=2000, DSMAX=0.2, bidirectional=True)

    # save results
    fname = '../results/gpe_2pop_c1.pkl'
    kwargs = {'k_pp': k_i_col}
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

    # continuation of eta_p
    #######################

    # step 1: codim 1 investigation
    c2_b1_sols, c2_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params, name='c2:eta_p',
                                   NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=starting_cont, NMX=6000, DSMAX=0.1,
                                   bidirectional=True, UZR={2: [2.0]})

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
    c2_b1_fp6_sols, c2_b1_fp6_cont = a.run(starting_point='LP1', c='qif2', ICP=[20, 3], NPAR=n_params,
                                           name='c2:k_p/eta_a', NDIM=n_dim, RL0=0.1, RL1=10.0, origin=c2_b1_cont,
                                           NMX=6000, DSMAX=0.5, bidirectional=True, STOP={'CP3'})

    # step 3: switch branch to zero-hopf bifurcations identified in step 2
    c2_b1_zh1_sols, c2_b1_zh1_cont = a.run(starting_point='ZH1', c='qif2b', ICP=[20, 11], NPAR=n_params,
                                           name='c2:k_p/eta_a/zh1', NDIM=n_dim, RL0=0.1, RL1=10.0, NMX=600, DSMAX=0.5,
                                           origin=c2_b1_fp6_cont, STOP={})

    starting_point = 'UZ3'
    starting_cont = c2_b1_cont

    # continuation of eta_a
    #######################

    # step 1: codim 1 investigation
    c2_b2_sols, c2_b2_cont = a.run(starting_point=starting_point, c='qif', ICP=3, NPAR=n_params, name='c2:eta_a',
                                   NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=starting_cont, NMX=6000, DSMAX=0.1,
                                   bidirectional=True)

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
                                   bidirectional=True, UZR={2: [3.0]})

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
