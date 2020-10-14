from pyrates.utility.pyauto import PyAuto, codim2_search
import numpy as np
import matplotlib.pyplot as plt
"""Bifurcation analysis of STN-GPe model with two GPe populations (arkypallidal and prototypical) and 
gamma-distributed axonal delays and bi-exponential synapses."""

# config
n_dim = 37
n_params = 29
a = PyAuto("auto_files", auto_dir="~/PycharmProjects/auto-07p")

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
                                   NPAR=n_params, RL0=-8.0, RL1=8.0, NMX=4000, DSMAX=0.1, name='c1:eta_p',
                                   bidirectional=True)

    # limit cycle continuations
    c1_b3_lc_sols, c1_b3_lc_cont = a.run(starting_point='HB1', origin=c1_b3_cont, c='qif2b', ICP=[2, 11],
                                         NDIM=n_dim, NPAR=n_params, RL0=-8.0, RL1=8.0, NMX=4000, DSMAX=0.2,
                                         name='c1:eta_p:lc1')
    c1_b3_lc2_sols, c1_b3_lc2_cont = a.run(starting_point='HB2', origin=c1_b3_cont, c='qif2b', ICP=[2, 11],
                                           NDIM=n_dim, NPAR=n_params, RL0=-8.0, RL1=8.0, NMX=4000, DSMAX=0.2,
                                           name='c1:eta_p:lc2')

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
                                   NPAR=n_params, RL0=-3.0, RL1=8.0, NMX=6000, DSMAX=0.1,
                                   name='c1.2:eta_p', bidirectional=True)

    # limit cycle continuation
    c1_b5_lc_sols, c1_b5_lc_cont = a.run(starting_point='HB1', origin=c1_b5_cont, c='qif2b', ICP=[2, 11],
                                         NDIM=n_dim, NPAR=n_params, RL0=-3.0, RL1=8.0, NMX=4000, DSMAX=0.2,
                                         name='c1.2:eta_p:lc1')
    c1_b5_lc2_sols, c1_b5_lc2_cont = a.run(starting_point='HB2', origin=c1_b5_cont, c='qif2b', ICP=[2, 11],
                                           NDIM=n_dim, NPAR=n_params, RL0=-3.0, RL1=8.0, NMX=4000, DSMAX=0.2,
                                           name='c1.2:eta_p:lc2')

    # continuation of hopf curve
    c1_b5_2d1_sols, c1_b5_2d1_cont = a.run(starting_point='HB1', origin=c1_b5_cont, c='qif2', ICP=[5, 10], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.1,
                                           name='c1.2:k_pe/k_pa:hb1', bidirectional=True)
    c1_b5_2d2_sols, c1_b5_2d2_cont = a.run(starting_point='HB2', origin=c1_b5_cont, c='qif2', ICP=[5, 10], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.1,
                                           name='c1.2:k_pe/k_pa:hb2', bidirectional=True)
    c1_b5_2d3_sols, c1_b5_2d3_cont = a.run(starting_point='HB1', origin=c1_b5_cont, c='qif2', ICP=[6, 3], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.1,
                                           name='c1.2:k_ae/eta_a:hb1', bidirectional=True, UZR={6: [8.0]})
    c1_b5_2d4_sols, c1_b5_2d4_cont = a.run(starting_point='HB2', origin=c1_b5_cont, c='qif2', ICP=[6, 3], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.1,
                                           name='c1.2:k_ae/eta_a:hb2', bidirectional=True, UZR={6: [8.0]})

    # investigation of impact of STN -> GPe-a coupling
    ##################################################

    # eta_p continuation
    c1_b6_sols, c1_b6_cont = a.run(starting_point='UZ1', origin=c1_b5_2d3_cont, c='qif', ICP=2, NDIM=n_dim,
                                   NPAR=n_params, RL0=-10.0, RL1=10.0, NMX=2000, DSMAX=0.1, UZR={2: [3.0]},
                                   name='c1.3:eta_p')

    # eta_a continuation
    c1_b7_sols, c1_b7_cont = a.run(starting_point='UZ1', origin=c1_b6_cont, c='qif', ICP=3, NDIM=n_dim,
                                   NPAR=n_params, RL0=-10.0, RL1=10.0, NMX=6000, DSMAX=0.1, bidirectional=True,
                                   name='c1.3:eta_a')

    # k_gp continuation
    c1_b8_sols, c1_b8_cont = a.run(starting_point='UZ1', origin=c1_b6_cont, c='qif', ICP=22, NDIM=n_dim,
                                   NPAR=n_params, RL0=0.0, RL1=8.5, NMX=6000, DSMAX=0.1, name='c1.3:k_gp')

    # limit cycle continuations
    c1_b7_lc_sols, c1_b7_lc_cont = a.run(starting_point='HB1', origin=c1_b7_cont, c='qif2b', ICP=[3, 11], NDIM=n_dim,
                                         NPAR=n_params, RL0=-10.0, RL1=10.0, NMX=2000, DSMAX=0.2, name='c1.3:eta_a:lc',
                                         STOP=['BP1'])
    c1_b8_lc_sols, c1_b8_lc_cont = a.run(starting_point='HB1', origin=c1_b8_cont, c='qif2b', ICP=[22, 11], NDIM=n_dim,
                                         NPAR=n_params, RL0=0.0, RL1=8.5, NMX=2000, DSMAX=0.2, name='c1.3:k_gp:lc',
                                         STOP=['BP1'])

    # continuation of fold curves
    c1_b7_2d1_sols, c1_b7_2d1_cont = a.run(starting_point='LP1', origin=c1_b7_cont, c='qif2', ICP=[9, 22], NDIM=n_dim,
                                           NPAR=n_params, RL0=-20.0, RL1=20.0, NMX=6000, DSMAX=0.1,
                                           name='c1.3:k_ap/k_gp:lp', bidirectional=True)
    c1_b7_2d2_sols, c1_b7_2d2_cont = a.run(starting_point='LP1', origin=c1_b7_cont, c='qif2', ICP=[5, 22], NDIM=n_dim,
                                           NPAR=n_params, RL0=-20.0, RL1=20.0, NMX=6000, DSMAX=0.1,
                                           name='c1.3:k_ep/k_gp:lp', bidirectional=True)
    c1_b7_2d3_sols, c1_b7_2d3_cont = a.run(starting_point='LP1', origin=c1_b7_cont, c='qif2', ICP=[2, 3], NDIM=n_dim,
                                           NPAR=n_params, RL0=-20.0, RL1=20.0, NMX=6000, DSMAX=0.1,
                                           name='c1.3:eta_p/eta_a:lp', bidirectional=True)

    # continuation of hopf curves
    c1_b8_2d1_sols, c1_b8_2d1_cont = a.run(starting_point='HB1', origin=c1_b8_cont, c='qif2', ICP=[9, 22], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.1,
                                           name='c1.3:k_ap/k_gp:hb', bidirectional=True)
    c1_b8_2d2_sols, c1_b8_2d2_cont = a.run(starting_point='HB1', origin=c1_b8_cont, c='qif2', ICP=[5, 22], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.1,
                                           name='c1.3:k_ep/k_gp:hb', bidirectional=True)
    c1_b8_2d3_sols, c1_b8_2d3_cont = a.run(starting_point='HB1', origin=c1_b8_cont, c='qif2', ICP=[2, 3], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.1,
                                           name='c1.3:eta_p/eta_a:hb', bidirectional=True)

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
                                   NPAR=n_params, RL0=-5.0, RL1=10.0, NMX=4000, DSMAX=0.1, name='c2:eta_p',
                                   bidirectional=True)

    # limit cycle continuations
    c2_b3_lc_sols, c2_b3_lc_cont = a.run(starting_point='HB1', origin=c2_b3_cont, c='qif2b', ICP=[2, 11],
                                         NDIM=n_dim, NPAR=n_params, RL0=-5.0, RL1=10.0, NMX=4000, DSMAX=0.2,
                                         name='c2:eta_p:lc1')
    c2_b3_lc2_sols, c2_b3_lc2_cont = a.run(starting_point='HB2', origin=c2_b3_cont, c='qif2b', ICP=[2, 11],
                                           NDIM=n_dim, NPAR=n_params, RL0=-5.0, RL1=10.0, NMX=4000, DSMAX=0.2,
                                           name='c2:eta_p:lc2')

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
                                   NPAR=n_params, RL0=-5.0, RL1=10.0, NMX=6000, DSMAX=0.1,
                                   name='c2.2:eta_p', bidirectional=True)

    # limit cycle continuation
    c2_b5_lc_sols, c2_b5_lc_cont = a.run(starting_point='HB1', origin=c2_b5_cont, c='qif2b', ICP=[2, 11],
                                         NDIM=n_dim, NPAR=n_params, RL0=-5.0, RL1=10.0, NMX=4000, DSMAX=0.2,
                                         name='c2.2:eta_p:lc1')
    c2_b5_lc2_sols, c2_b5_lc2_cont = a.run(starting_point='HB2', origin=c2_b5_cont, c='qif2b', ICP=[2, 11],
                                           NDIM=n_dim, NPAR=n_params, RL0=-5.0, RL1=10.0, NMX=4000, DSMAX=0.2,
                                           name='c2.2:eta_p:lc2')

    # continuation of hopf curve
    c2_b5_2d1_sols, c2_b5_2d1_cont = a.run(starting_point='HB1', origin=c2_b5_cont, c='qif2', ICP=[5, 10], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.1,
                                           name='c2.2:k_pe/k_pa:hb1', bidirectional=True)
    c2_b5_2d2_sols, c2_b5_2d2_cont = a.run(starting_point='HB2', origin=c2_b5_cont, c='qif2', ICP=[5, 10], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.1,
                                           name='c2.2:k_pe/k_pa:hb2', bidirectional=True)
    c2_b5_2d3_sols, c2_b5_2d3_cont = a.run(starting_point='HB1', origin=c2_b5_cont, c='qif2', ICP=[6, 3], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.1,
                                           name='c2.2:k_ae/eta_a:hb1', bidirectional=True, UZR={6: [8.0]})
    c2_b5_2d4_sols, c2_b5_2d4_cont = a.run(starting_point='HB2', origin=c2_b5_cont, c='qif2', ICP=[6, 3], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.1,
                                           name='c2.2:k_ae/eta_a:hb2', bidirectional=True, UZR={6: [8.0]})

    # investigation of impact of STN -> GPe-a coupling
    ##################################################

    # eta_p continuation
    c2_b6_sols, c2_b6_cont = a.run(starting_point='UZ1', origin=c2_b5_2d3_cont, c='qif', ICP=2, NDIM=n_dim,
                                   NPAR=n_params, RL0=-10.0, RL1=10.0, NMX=2000, DSMAX=0.1, UZR={2: [6.0]},
                                   name='c2.3:eta_p')

    # k_ap continuation
    c2_b7_sols, c2_b7_cont = a.run(starting_point='UZ1', origin=c2_b6_cont, c='qif', ICP=9, NDIM=n_dim,
                                   NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.1, name='c2.3:k_ap')

    # k_gp continuation
    c2_b8_sols, c2_b8_cont = a.run(starting_point='UZ1', origin=c2_b6_cont, c='qif', ICP=22, NDIM=n_dim,
                                   NPAR=n_params, RL0=0.0, RL1=12.0, NMX=6000, DSMAX=0.1, name='c2.3:k_gp')

    # limit cycle continuations
    c2_b7_lc_sols, c2_b7_lc_cont = a.run(starting_point='HB1', origin=c2_b7_cont, c='qif2b', ICP=[9, 11],
                                         NDIM=n_dim, NPAR=n_params, RL0=0.0, RL1=6.0, NMX=4000, DSMAX=0.2,
                                         name='c2.3:k_ap:lc', STOP=['BP1'])
    c2_b8_lc_sols, c2_b8_lc_cont = a.run(starting_point='HB1', origin=c2_b8_cont, c='qif2b', ICP=[22, 11],
                                         NDIM=n_dim, NPAR=n_params, RL0=0.0, RL1=12.0, NMX=4000, DSMAX=0.2,
                                         name='c2.3:k_gp:lc', STOP=['BP1'])

    # continuation of hopf curves
    c2_b7_2d1_sols, c2_b7_2d1_cont = a.run(starting_point='HB1', origin=c2_b7_cont, c='qif2', ICP=[5, 2], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.1,
                                           name='c2.3:k_pe/eta_p', bidirectional=True)
    c2_b7_2d2_sols, c2_b7_2d2_cont = a.run(starting_point='HB1', origin=c2_b7_cont, c='qif2', ICP=[6, 9], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.1,
                                           name='c2.3:k_ae/k_ap', bidirectional=True)
    c2_b7_2d3_sols, c2_b7_2d3_cont = a.run(starting_point='HB1', origin=c2_b7_cont, c='qif2', ICP=[22, 9], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.1,
                                           name='c2.3:k_gp/k_ap', bidirectional=True)

    c2_b8_2d1_sols, c2_b8_2d1_cont = a.run(starting_point='HB1', origin=c2_b8_cont, c='qif2', ICP=[9, 22], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.1,
                                           name='c2.3:k_ap/k_gp', bidirectional=True)
    c2_b8_2d2_sols, c2_b8_2d2_cont = a.run(starting_point='HB1', origin=c2_b8_cont, c='qif2', ICP=[5, 22], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.1,
                                           name='c2.3:k_pe/k_gp', bidirectional=True)
    c2_b8_2d3_sols, c2_b8_2d3_cont = a.run(starting_point='HB1', origin=c2_b8_cont, c='qif2', ICP=[22, 2], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.1,
                                           name='c2.3:k_gp/eta_p', bidirectional=True)

else:
    raise ValueError('Exactly one of the two conditions must be set to true: c1 or c2.')

# save results
kwargs = dict()
a.to_file(fname, **kwargs)
