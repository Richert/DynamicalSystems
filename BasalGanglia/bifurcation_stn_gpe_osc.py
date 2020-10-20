from pyrates.utility.pyauto import PyAuto, get_from_solutions
import numpy as np
import matplotlib.pyplot as plt
"""Bifurcation analysis of STN-GPe model with two GPe populations (arkypallidal and prototypical) and 
gamma-distributed axonal delays and bi-exponential synapses."""

# config
n_dim = 37
n_params = 29
a = PyAuto("auto_files", auto_dir="~/PycharmProjects/auto-07p")
kwargs = dict()

# choose condition
c1 = False  # bi-stable
c2 = True   # oscillatory

#########################
# initial continuations #
#########################

# continuation in time
t_sols, t_cont = a.run(e='stn_gpe_final', c='ivp', ICP=14, NMX=1000000, name='t', UZR={14: 1000.0}, STOP={'UZ1'},
                       NDIM=n_dim, NPAR=n_params)
starting_point = 'UZ1'
starting_cont = t_cont

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

# step 3: continuation of k_pe
s2_sols, s2_cont = a.run(starting_point=starting_point, c='qif', ICP=5, NPAR=n_params, name='k_pe', NDIM=n_dim,
                         RL0=0.0, RL1=10.0, origin=starting_cont, NMX=8000, DSMAX=0.1,
                         UZR={5: [2.0, 4.0, 6.0, 8.0, 10.0]}, STOP={})
starting_point = 'UZ4'
starting_cont = s2_cont

# step 4: continuation of k_ep
s3_sols, s3_cont = a.run(starting_point=starting_point, c='qif', ICP=7, NPAR=n_params, name='k_ep', NDIM=n_dim,
                         RL0=0.0, RL1=20.0, origin=starting_cont, NMX=8000, DSMAX=0.1,
                         UZR={7: [6.0, 8.0, 10.0, 12.0, 14.0, 16.0]}, STOP={})
starting_point = 'UZ2'
starting_cont = s3_cont

# step 5: continuation of k_pa
s4_sols, s4_cont = a.run(starting_point=starting_point, c='qif', ICP=10, NPAR=n_params, name='k_pa', NDIM=n_dim,
                         RL0=-0.1, RL1=10.0, origin=starting_cont, NMX=2000, DSMAX=0.1, DS='-',
                         UZR={10: [0.0]}, STOP={'UZ1'})
starting_point = 'UZ1'
starting_cont = s4_cont

# step 6: choose relative strength of inter- vs. intra-population coupling inside GPe
s5_sols, s5_cont = a.run(starting_point=starting_point, c='qif', ICP=24, NPAR=n_params, name='k_i', NDIM=n_dim,
                         RL0=0.8, RL1=2.0, origin=starting_cont, NMX=2000, DSMAX=0.1, bidirectional=True,
                         UZR={24: [0.9, 1.8]}, STOP={})

##############################################################################
# c1: investigation of STN-GPe behavior near the bi-stable regime of the GPe #
##############################################################################

if c1:

    fname = '../results/stn_gpe_osc_c1.pkl'
    starting_point = 'UZ2'
    starting_cont = s5_cont

    # preparation of healthy state
    ##############################

    # continuation of eta_p
    c1_b1_sols, c1_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params,
                                   name='c1:eta_p', NDIM=n_dim, RL0=-10.0, RL1=20.0, origin=starting_cont,
                                   NMX=4000, DSMAX=0.1, UZR={2: [0.5]}, bidirectional=True)
    starting_point = 'UZ1'
    starting_cont = c1_b1_cont

    # continuation of eta_e
    c1_b2_sols, c1_b2_cont = a.run(starting_point=starting_point, c='qif', ICP=1, NPAR=n_params,
                                   name='c1:eta_e', NDIM=n_dim, RL0=-10.0, RL1=10.0, origin=starting_cont,
                                   NMX=4000, DSMAX=0.1, UZR={1: [3.0]})
    starting_point = 'UZ1'
    starting_cont = c1_b2_cont

    # continuation of parkinsonian parameters
    #########################################

    # continuation of eta_p output
    c1_b3_sols, c1_b3_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif', ICP=2, NDIM=n_dim,
                                   NPAR=n_params, RL0=-8.0, RL1=8.0, NMX=4000, DSMAX=0.05, name='c1:eta_p',
                                   bidirectional=True)

    # limit cycle continuations
    # c1_b3_lc_sols, c1_b3_lc_cont = a.run(starting_point='HB1', origin=c1_b3_cont, c='qif2b', ICP=[2, 11],
    #                                      NDIM=n_dim, NPAR=n_params, RL0=-8.0, RL1=8.0, NMX=4000, DSMAX=0.1,
    #                                      name='c1:eta_p:lc1')
    # c1_b3_lc2_sols, c1_b3_lc2_cont = a.run(starting_point='HB2', origin=c1_b3_cont, c='qif2b', ICP=[2, 11],
    #                                        NDIM=n_dim, NPAR=n_params, RL0=-8.0, RL1=8.0, NMX=4000, DSMAX=0.1,
    #                                        name='c1:eta_p:lc2')

    # continuation of hopf curves
    c1_b3_2d1_sols, c1_b3_2d1_cont = a.run(starting_point='HB1', origin=c1_b3_cont, c='qif2', ICP=[5, 2], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.1,
                                           name='c1:k_pe/eta_p:hb1', bidirectional=True)
    c1_b3_2d2_sols, c1_b3_2d2_cont = a.run(starting_point='HB2', origin=c1_b3_cont, c='qif2', ICP=[5, 2], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.1,
                                           name='c1:k_pe/eta_p:hb2', bidirectional=True)
    c1_b3_2d3_sols, c1_b3_2d3_cont = a.run(starting_point='HB1', origin=c1_b3_cont, c='qif2', ICP=[10, 2], NDIM=n_dim,
                                           NPAR=n_params, RL1=10.0, NMX=6000, DSMAX=0.1, UZR={10: [1.0]},
                                           name='c1:k_pa/eta_p:hb1')
    c1_b3_2d4_sols, c1_b3_2d4_cont = a.run(starting_point='HB2', origin=c1_b3_cont, c='qif2', ICP=[10, 2], NDIM=n_dim,
                                           NPAR=n_params, RL1=10.0, NMX=6000, DSMAX=0.1, UZR={10: [1.0]},
                                           name='c1:k_pa/eta_p:hb2')

    # investigation of GPe-a impact on STN-GPe oscillations
    #######################################################

    # continuation away from hopf bifurcation
    c1_b4_sols, c1_b4_cont = a.run(starting_point='UZ1', origin=c1_b3_2d3_cont, c='qif', ICP=3, NDIM=n_dim,
                                   NPAR=n_params, RL0=-10.0, RL1=10.0, NMX=6000, DSMAX=0.1, UZR={3: [2.0]},
                                   name='c1.2:eta_a', bidirectional=True)

    # eta_p continuation
    c1_b5_sols, c1_b5_cont = a.run(starting_point='UZ1', origin=c1_b4_cont, c='qif', ICP=2, NDIM=n_dim,
                                   NPAR=n_params, RL0=-3.0, RL1=8.0, NMX=6000, DSMAX=0.05,
                                   name='c1.2:eta_p', bidirectional=True)

    # limit cycle continuation
    # c1_b5_lc_sols, c1_b5_lc_cont = a.run(starting_point='HB1', origin=c1_b5_cont, c='qif2b', ICP=[2, 11],
    #                                      NDIM=n_dim, NPAR=n_params, RL0=-3.0, RL1=8.0, NMX=4000, DSMAX=0.1,
    #                                      name='c1.2:eta_p:lc1')
    # c1_b5_lc2_sols, c1_b5_lc2_cont = a.run(starting_point='HB2', origin=c1_b5_cont, c='qif2b', ICP=[2, 11],
    #                                        NDIM=n_dim, NPAR=n_params, RL0=-3.0, RL1=8.0, NMX=4000, DSMAX=0.1,
    #                                        name='c1.2:eta_p:lc2')

    # continuation of hopf curve
    c1_b5_2d1_sols, c1_b5_2d1_cont = a.run(starting_point='HB1', origin=c1_b5_cont, c='qif2', ICP=[10, 2], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.1,
                                           name='c1.2:k_pa/eta_p:hb1', bidirectional=True)
    c1_b5_2d2_sols, c1_b5_2d2_cont = a.run(starting_point='HB2', origin=c1_b5_cont, c='qif2', ICP=[10, 2], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.1,
                                           name='c1.2:k_pa/eta_p:hb2', bidirectional=True)
    c1_b5_2d3_sols, c1_b5_2d3_cont = a.run(starting_point='HB1', origin=c1_b5_cont, c='qif2', ICP=[6, 2], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.1,
                                           name='c1.2:k_ae/eta_p:hb1', bidirectional=True, UZR={6: [4.0]})
    c1_b5_2d4_sols, c1_b5_2d4_cont = a.run(starting_point='HB2', origin=c1_b5_cont, c='qif2', ICP=[6, 2], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.1,
                                           name='c1.2:k_ae/eta_p:hb2', bidirectional=True, UZR={6: [4.0]})

    # investigation of impact of STN -> GPe-a coupling
    ##################################################

    # eta_p continuation
    c1_b6_sols, c1_b6_cont = a.run(starting_point='UZ1', origin=c1_b5_2d4_cont, c='qif', ICP=2, NDIM=n_dim,
                                   NPAR=n_params, RL0=-20.0, RL1=20.0, NMX=4000, DSMAX=0.1, UZR={2: [3.5]},
                                   name='c1.3:eta_p', bidirectional=True)

    # k_gp continuation
    c1_b7_sols, c1_b7_cont = a.run(starting_point='UZ1', origin=c1_b6_cont, c='qif', ICP=22, NDIM=n_dim,
                                   NPAR=n_params, RL0=0.0, RL1=100.0, NMX=8000, DSMAX=0.05, name='c1.3:k_gp',
                                   bidirectional=True, UZR={22: [1.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 18.0, 20.0,
                                                                 23.0, 30.0, 35.0]})

    # limit cycle continuations
    # c1_b7_lc_sols, c1_b7_lc_cont = a.run(starting_point='HB1', origin=c1_b7_cont, c='qif2b', ICP=[22, 11], NDIM=n_dim,
    #                                      NPAR=n_params, RL0=0.0, RL1=9.0, NMX=2000, DSMAX=0.1, name='c1.3:k_gp:lc',
    #                                      STOP=['BP1'])
    i = 0
    for key, s in c1_b7_sols.items():
        if 'UZ' in s['bifurcation']:
            i += 1
            s_tmp, c_tmp = a.run(starting_point=f'UZ{i}', origin=c1_b7_cont, c='qif', ICP=5, NDIM=n_dim,
                                 NPAR=n_params, RL0=0.0, RL1=10.0, NMX=8000, DSMAX=0.05, name=f'c1.3:k_pe_{i}',
                                 bidirectional=True)
            j = 0
            for bf in get_from_solutions(['bifurcation'], solutions=s_tmp):
                if 'HB' in bf:
                    j += 1
                    a.run(starting_point=f'HB{j}', origin=c_tmp, c='qif2b', ICP=[5, 11], NDIM=n_dim,
                          NPAR=n_params, RL0=0.0, RL1=10.0, NMX=2000, DSMAX=0.1, name=f'c1.3:k_pe_{i}:lc{j}',
                          STOP=['BP1', 'LP3'])

    # continuation of hopf curves
    c1_b7_2d1_sols, c1_b7_2d1_cont = a.run(starting_point='HB1', origin=c1_b7_cont, c='qif2', ICP=[5, 22], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=15.0, NMX=8000, DSMAX=0.2,
                                           name='c1.3:k_pe/k_gp', bidirectional=True)
    c1_b7_2d2_sols, c1_b7_2d2_cont = a.run(starting_point='HB1', origin=c1_b7_cont, c='qif2', ICP=[2, 22], NDIM=n_dim,
                                           NPAR=n_params, RL0=-10.0, RL1=10.0, NMX=8000, DSMAX=0.2,
                                           name='c1.3:eta_p/k_gp', bidirectional=True)
    c1_b7_2d3_sols, c1_b7_2d3_cont = a.run(starting_point='HB1', origin=c1_b7_cont, c='qif2', ICP=[22, 24], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=15.0, NMX=8000, DSMAX=0.2,
                                           name='c1.3:k_gp/k_i', bidirectional=True)
    c1_b7_2d4_sols, c1_b7_2d4_cont = a.run(starting_point='HB1', origin=c1_b7_cont, c='qif2', ICP=[7, 22], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=15.0, NMX=8000, DSMAX=0.1,
                                           name='c1.3:k_ep/k_gp', bidirectional=True)

    # complete 2D  bifurcation diagram for k_pe x k_gp
    sols_tmp, cont_tmp = a.run(starting_point='ZH1', origin=c1_b7_2d1_cont, c='qif', ICP=5, NDIM=n_dim, NPAR=n_params,
                               RL0=0.0, RL1=15.0, NMX=4000, DSMAX=0.1, bidirectional=True, STOP=['LP1', 'HB1'])
    a.run(starting_point='HB1', origin=cont_tmp, c='qif2', ICP=[5, 22], NDIM=n_dim, NPAR=n_params, RL0=0.0, RL1=15.0,
          NMX=8000, DSMAX=0.2, name='c1.3:k_pe/k_gp:zh1', bidirectional=True)

    sols_tmp2, cont_tmp2 = a.run(starting_point='ZH2', origin=c1_b7_2d1_cont, c='qif', ICP=5, NDIM=n_dim, NPAR=n_params,
                                 RL0=0.0, RL1=15.0, NMX=4000, DSMAX=0.1, bidirectional=True, STOP=['LP1', 'HB1'])
    a.run(starting_point='HB2', origin=cont_tmp2, c='qif2', ICP=[5, 22], NDIM=n_dim, NPAR=n_params, RL0=0.0, RL1=15.0,
          NMX=8000, DSMAX=0.2, name='c1.3:k_pe/k_gp:zh2', bidirectional=True)

    sols_tmp3, cont_tmp3 = a.run(starting_point='ZH3', origin=c1_b7_2d1_cont, c='qif', ICP=5, NDIM=n_dim, NPAR=n_params,
                                 RL0=0.0, RL1=15.0, NMX=4000, DSMAX=0.1, bidirectional=True, STOP=['LP1', 'HB1'])
    a.run(starting_point='LP1', origin=cont_tmp3, c='qif2', ICP=[5, 22], NDIM=n_dim, NPAR=n_params, RL0=0.0, RL1=15.0,
          NMX=12000, DSMAX=0.2, name='c1.3:k_pe/k_gp:zh3', bidirectional=True)

################################################################################
# c2: investigation of STN-GPe behavior near the oscillatory regime of the GPe #
################################################################################

elif c2:

    fname = '../results/stn_gpe_osc_c2.pkl'
    starting_point = 'UZ1'
    starting_cont = s5_cont

    # preparation of healthy state
    ##############################

    # continuation of eta_p
    c2_b1_sols, c2_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params,
                                   name='c2:eta_p', NDIM=n_dim, RL0=-10.0, RL1=20.0, origin=starting_cont,
                                   NMX=4000, DSMAX=0.1, UZR={2: [4.0]})
    starting_point = 'UZ1'
    starting_cont = c2_b1_cont

    # continuation of eta_e
    c2_b2_sols, c2_b2_cont = a.run(starting_point=starting_point, c='qif', ICP=1, NPAR=n_params,
                                   name='c2:eta_e', NDIM=n_dim, RL0=-10.0, RL1=10.0, origin=starting_cont,
                                   NMX=4000, DSMAX=0.1, UZR={1: [3.0]})
    starting_point = 'UZ1'
    starting_cont = c2_b2_cont

    # continuation of parkinsonian parameters
    #########################################

    # continuation of eta_p output
    c2_b3_sols, c2_b3_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif', ICP=2, NDIM=n_dim,
                                   NPAR=n_params, RL0=-5.0, RL1=10.0, NMX=4000, DSMAX=0.05, name='c2:eta_p',
                                   bidirectional=True)

    # limit cycle continuations
    # c2_b3_lc_sols, c2_b3_lc_cont = a.run(starting_point='HB1', origin=c2_b3_cont, c='qif2b', ICP=[2, 11],
    #                                      NDIM=n_dim, NPAR=n_params, RL0=-5.0, RL1=10.0, NMX=4000, DSMAX=0.1,
    #                                      name='c2:eta_p:lc1')
    # c2_b3_lc2_sols, c2_b3_lc2_cont = a.run(starting_point='HB2', origin=c2_b3_cont, c='qif2b', ICP=[2, 11],
    #                                        NDIM=n_dim, NPAR=n_params, RL0=-5.0, RL1=10.0, NMX=4000, DSMAX=0.1,
    #                                        name='c2:eta_p:lc2')

    # continuation of hopf curves
    c2_b3_2d1_sols, c2_b3_2d1_cont = a.run(starting_point='HB1', origin=c2_b3_cont, c='qif2', ICP=[5, 2], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.1,
                                           name='c2:k_pe/eta_p:hb1', bidirectional=True)
    c2_b3_2d2_sols, c2_b3_2d2_cont = a.run(starting_point='HB2', origin=c2_b3_cont, c='qif2', ICP=[5, 2], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.1,
                                           name='c2:k_pe/eta_p:hb2', bidirectional=True)
    c2_b3_2d3_sols, c2_b3_2d3_cont = a.run(starting_point='HB1', origin=c2_b3_cont, c='qif2', ICP=[10, 2], NDIM=n_dim,
                                           NPAR=n_params, RL1=10.0, NMX=6000, DSMAX=0.1, UZR={10: [1.0]},
                                           name='c2:k_pa/eta_p:hb1')
    c2_b3_2d4_sols, c2_b3_2d4_cont = a.run(starting_point='HB2', origin=c2_b3_cont, c='qif2', ICP=[10, 2], NDIM=n_dim,
                                           NPAR=n_params, RL1=10.0, NMX=6000, DSMAX=0.1, UZR={10: [1.0]},
                                           name='c2:k_pa/eta_p:hb2')

    # investigation of GPe-a impact on STN-GPe oscillations
    #######################################################

    # continuation away from hopf bifurcation
    c2_b4_sols, c2_b4_cont = a.run(starting_point='UZ1', origin=c2_b3_2d3_cont, c='qif', ICP=3, NDIM=n_dim,
                                   NPAR=n_params, RL0=-10.0, RL1=10.0, NMX=6000, DSMAX=0.1, UZR={3: [-3.0]},
                                   name='c2.2:eta_a', bidirectional=True)

    # eta_p continuation
    c2_b5_sols, c2_b5_cont = a.run(starting_point='UZ1', origin=c2_b4_cont, c='qif', ICP=2, NDIM=n_dim,
                                   NPAR=n_params, RL0=-10.0, RL1=25.0, NMX=8000, DSMAX=0.05,
                                   name='c2.2:eta_p', bidirectional=True)

    # limit cycle continuation
    # c2_b5_lc_sols, c2_b5_lc_cont = a.run(starting_point='HB1', origin=c2_b5_cont, c='qif2b', ICP=[2, 11],
    #                                      NDIM=n_dim, NPAR=n_params, RL0=-5.0, RL1=10.0, NMX=4000, DSMAX=0.1,
    #                                      name='c2.2:eta_p:lc1')
    # c2_b5_lc2_sols, c2_b5_lc2_cont = a.run(starting_point='HB2', origin=c2_b5_cont, c='qif2b', ICP=[2, 11],
    #                                        NDIM=n_dim, NPAR=n_params, RL0=-5.0, RL1=10.0, NMX=4000, DSMAX=0.1,
    #                                        name='c2.2:eta_p:lc2')

    # continuation of hopf curve
    c2_b5_2d1_sols, c2_b5_2d1_cont = a.run(starting_point='HB1', origin=c2_b5_cont, c='qif2', ICP=[10, 2], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.1,
                                           name='c2.2:k_pa/eta_p:hb1', bidirectional=True)
    c2_b5_2d2_sols, c2_b5_2d2_cont = a.run(starting_point='HB2', origin=c2_b5_cont, c='qif2', ICP=[10, 2], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.1,
                                           name='c2.2:k_pa/eta_p:hb2', bidirectional=True)
    c2_b5_2d3_sols, c2_b5_2d3_cont = a.run(starting_point='HB1', origin=c2_b5_cont, c='qif2', ICP=[6, 2], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.1,
                                           name='c2.2:k_ae/eta_p:hb1', bidirectional=True, UZR={6: [8.0]})
    c2_b5_2d4_sols, c2_b5_2d4_cont = a.run(starting_point='HB2', origin=c2_b5_cont, c='qif2', ICP=[6, 2], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.1,
                                           name='c2.2:k_ae/eta_p:hb2', bidirectional=True, UZR={6: [8.0]})

    # get limit cycle periods of 2D grid
    N = 101
    k_pa_sweep = np.linspace(0.0, 10.0, N)
    eta_p_sweep = np.linspace(-10, 25.0, N)
    period_solutions = np.zeros((N, N))

    s_tmp, c_tmp = a.run(starting_point='UZ1', origin=c2_b4_cont, c='qif', ICP=10, NDIM=n_dim,
                         NPAR=n_params, RL0=-0.1, RL1=10.0, NMX=8000, DSMAX=0.05,
                         name='c2.2:k_pa', bidirectional=True, UZR={10: k_pa_sweep})

    i = 0
    for s, s_info in s_tmp.items():
        if 'UZ' in s_info['bifurcation']:
            i += 1
            s_tmp2, c_tmp2 = a.run(starting_point=f'UZ{i}', c='qif', ICP=2, RL0=-10.1, RL1=25.1, DSMAX=0.05,
                                   origin=c_tmp, NMX=6000, bidirectional=True, NDIM=n_dim, NPAR=n_params)
            s_lc1, _ = a.run(starting_point='HB1', origin=c_tmp2, c='qif2b', ICP=[2, 11], UZR={2: eta_p_sweep}, STOP={},
                             DSMAX=0.5, NMX=2000, get_period=True, RL0=-10.1, RL1=25.1, NDIM=n_dim, NPAR=n_params)
            s_lc2, _ = a.run(starting_point='HB2', origin=c_tmp2, c='qif2b', ICP=[2, 11], UZR={2: eta_p_sweep}, STOP={},
                             DSMAX=0.5, NMX=2000, get_period=True, RL0=-10.1, RL1=25.1, NDIM=n_dim, NPAR=n_params)
            for s1 in s_lc1.values():
                if 'UZ' in s1['bifurcation']:
                    idx_c = np.argmin(np.abs(np.round(s1['PAR(2)'], decimals=3) - eta_p_sweep))
                    idx_r = np.argmin(np.abs(np.round(s1['PAR(10)'], decimals=3) - k_pa_sweep))
                    period_solutions[idx_r, idx_c] = s1['period']
            for s2 in s_lc2.values():
                if 'UZ' in s2['bifurcation']:
                    idx_c = np.argmin(np.abs(np.round(s2['PAR(2)'], decimals=3) - eta_p_sweep))
                    idx_r = np.argmin(np.abs(np.round(s2['PAR(10)'], decimals=3) - k_pa_sweep))
                    period_solutions[idx_r, idx_c] = s2['period']

    # save results
    kwargs['period_solutions'] = period_solutions
    kwargs['eta_p'] = eta_p_sweep
    kwargs['k_pa'] = k_pa_sweep
    a.to_file(fname, **kwargs)

    # investigation of impact of STN -> GPe-a coupling
    ##################################################

    # eta_p continuation
    c2_b6_sols, c2_b6_cont = a.run(starting_point='UZ1', origin=c2_b5_2d4_cont, c='qif', ICP=2, NDIM=n_dim,
                                   NPAR=n_params, RL0=-10.0, RL1=10.0, NMX=2000, DSMAX=0.1, UZR={2: [6.0]},
                                   name='c2.3:eta_p', bidirectional=True)

    # k_ap continuation
    c2_b7_sols, c2_b7_cont = a.run(starting_point='UZ1', origin=c2_b6_cont, c='qif', ICP=9, NDIM=n_dim,
                                   NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.05, name='c2.3:k_ap',
                                   bidirectional=True)

    # limit cycle continuations
    # c2_b7_lc_sols, c2_b7_lc_cont = a.run(starting_point='HB1', origin=c2_b7_cont, c='qif2b', ICP=[9, 11],
    #                                      NDIM=n_dim, NPAR=n_params, RL0=0.0, RL1=8.0, NMX=4000, DSMAX=0.1,
    #                                      name='c2.3:k_ap:lc', STOP=['BP1'])
    # c2_b8_lc_sols, c2_b8_lc_cont = a.run(starting_point='HB2', origin=c2_b8_cont, c='qif2b', ICP=[22, 11],
    #                                      NDIM=n_dim, NPAR=n_params, RL0=1.0, RL1=11.0, NMX=4000, DSMAX=0.1,
    #                                      name='c2.3:k_gp:lc1', STOP=['BP1'])
    # c2_b8_lc2_sols, c2_b8_lc2_cont = a.run(starting_point='HB3', origin=c2_b8_cont, c='qif2b', ICP=[22, 11],
    #                                        NDIM=n_dim, NPAR=n_params, RL0=1.0, RL1=11.0, NMX=4000, DSMAX=0.1,
    #                                        name='c2.3:k_gp:lc2', STOP=['BP1'])

    # continuation of hopf curves
    c2_b7_2d1_sols, c2_b7_2d1_cont = a.run(starting_point='HB1', origin=c2_b7_cont, c='qif2', ICP=[5, 9], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.2,
                                           name='c2.3:k_pe/k_ap', bidirectional=True)
    c2_b7_2d2_sols, c2_b7_2d2_cont = a.run(starting_point='HB1', origin=c2_b7_cont, c='qif2', ICP=[6, 9], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.2,
                                           name='c2.3:k_ae/k_ap', bidirectional=True)
    c2_b7_2d3_sols, c2_b7_2d3_cont = a.run(starting_point='HB1', origin=c2_b7_cont, c='qif2', ICP=[10, 9], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.2,
                                           name='c2.3:k_pa/k_ap', bidirectional=True)
    c2_b7_2d4_sols, c2_b7_2d4_cont = a.run(starting_point='HB1', origin=c2_b7_cont, c='qif2', ICP=[8, 9], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.2,
                                           name='c2.3:k_pp/k_ap', bidirectional=True)
    c2_b7_2d5_sols, c2_b7_2d5_cont = a.run(starting_point='HB1', origin=c2_b7_cont, c='qif2', ICP=[7, 9], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=16.0, NMX=6000, DSMAX=0.2,
                                           name='c2.3:k_ep/k_ap', bidirectional=True)

    # c2_b8_2d1_sols, c2_b8_2d1_cont = a.run(starting_point='HB3', origin=c2_b8_cont, c='qif2', ICP=[9, 22], NDIM=n_dim,
    #                                        NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.2,
    #                                        name='c2.3:k_ap/k_gp', bidirectional=True)
    # c2_b8_2d2_sols, c2_b8_2d2_cont = a.run(starting_point='HB3', origin=c2_b8_cont, c='qif2', ICP=[5, 22], NDIM=n_dim,
    #                                        NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.2,
    #                                        name='c2.3:k_pe/k_gp', bidirectional=True)
    # c2_b8_2d3_sols, c2_b8_2d3_cont = a.run(starting_point='HB3', origin=c2_b8_cont, c='qif2', ICP=[6, 22], NDIM=n_dim,
    #                                        NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.2,
    #                                        name='c2.3:k_ae/k_gp', bidirectional=True)

else:
    raise ValueError('Exactly one of the two conditions must be set to true: c1 or c2.')

# save results
a.to_file(fname, **kwargs)
