from pyrates.utility.pyauto import PyAuto, get_from_solutions
import numpy as np
import sys

"""Bifurcation analysis of STN-GPe model with two GPe populations (arkypallidal and prototypical) and 
gamma-distributed axonal delays and bi-exponential synapses."""

path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"

# config
n_dim = 37
n_params = 29
a = PyAuto("auto_files", auto_dir=auto_dir)
kwargs = dict()

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

################################################################################
# investigation of STN-GPe behavior near the oscillatory regime of the GPe #
################################################################################

fname = '../results/stn_gpe_osc.pkl'
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
c2_b3_lc_sols, c2_b3_lc_cont = a.run(starting_point='HB1', origin=c2_b3_cont, c='qif2b', ICP=[2, 11],
                                     NDIM=n_dim, NPAR=n_params, RL0=-5.0, RL1=10.0, NMX=4000, DSMAX=0.1,
                                     name='c2:eta_p:lc1')
c2_b3_lc2_sols, c2_b3_lc2_cont = a.run(starting_point='HB2', origin=c2_b3_cont, c='qif2b', ICP=[2, 11],
                                       NDIM=n_dim, NPAR=n_params, RL0=-5.0, RL1=10.0, NMX=4000, DSMAX=0.1,
                                       name='c2:eta_p:lc2')

# continuation of hopf curves
c2_b3_2d1_sols, c2_b3_2d1_cont = a.run(starting_point='HB1', origin=c2_b3_cont, c='qif2', ICP=[5, 2], NDIM=n_dim,
                                       NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.1,
                                       name='c2:k_pe/eta_p:hb1', bidirectional=True)
c2_b3_2d2_sols, c2_b3_2d2_cont = a.run(starting_point='HB2', origin=c2_b3_cont, c='qif2', ICP=[5, 2], NDIM=n_dim,
                                       NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.1,
                                       name='c2:k_pe/eta_p:hb2', bidirectional=True)
c2_b3_2d3_sols, c2_b3_2d3_cont = a.run(starting_point='HB1', origin=c2_b3_cont, c='qif2', ICP=[10, 2], NDIM=n_dim,
                                       NPAR=n_params, RL1=15.0, NMX=6000, DSMAX=0.1, UZR={2: [6.0]},
                                       name='c2:k_pa/eta_p:hb1')
c2_b3_2d4_sols, c2_b3_2d4_cont = a.run(starting_point='HB2', origin=c2_b3_cont, c='qif2', ICP=[10, 2], NDIM=n_dim,
                                       NPAR=n_params, RL1=15.0, NMX=6000, DSMAX=0.1, UZR={2: [6.0]},
                                       name='c2:k_pa/eta_p:hb2')
c2_b3_2d5_sols, c2_b3_2d5_cont = a.run(starting_point='HB1', origin=c2_b3_cont, c='qif2', ICP=[8, 2], NDIM=n_dim,
                                       NPAR=n_params, RL0=0.0, RL1=15.0, NMX=6000, DSMAX=0.1,
                                       name='c2:k_pp/eta_p:hb1', bidirectional=True)
c2_b3_2d6_sols, c2_b3_2d6_cont = a.run(starting_point='HB2', origin=c2_b3_cont, c='qif2', ICP=[8, 2], NDIM=n_dim,
                                       NPAR=n_params, RL0=0.0, RL1=15.0, NMX=6000, DSMAX=0.1,
                                       name='c2:k_pp/eta_p:hb2', bidirectional=True)

# complete 2D  bifurcation diagram for k_pp x eta_p
sols_tmp, cont_tmp = a.run(starting_point='ZH2', origin=c2_b3_2d6_cont, c='qif', ICP=2, NDIM=n_dim, NPAR=n_params,
                           RL0=-50.0, RL1=8.0, NMX=4000, DSMAX=0.1, bidirectional=True, STOP=['LP1', 'HB1'])
a.run(starting_point='HB1', origin=cont_tmp, c='qif2', ICP=[8, 2], NDIM=n_dim, NPAR=n_params, RL0=0.0, RL1=15.0,
      NMX=8000, DSMAX=0.1, name='c2:k_pp/eta_p:zh1', bidirectional=True)

# investigation of GPe-a impact on STN-GPe oscillations
#######################################################

# continuation away from hopf bifurcation
c2_b4_sols, c2_b4_cont = a.run(starting_point='UZ1', origin=c2_b3_2d3_cont, c='qif', ICP=3, NDIM=n_dim,
                               NPAR=n_params, RL0=-10.0, RL1=10.0, NMX=6000, DSMAX=0.1, UZR={3: [-3.0]},
                               name='c2.2:eta_a', bidirectional=True)

# k_pa continuation
c2_b5_sols, c2_b5_cont = a.run(starting_point='UZ1', origin=c2_b4_cont, c='qif', ICP=10, NDIM=n_dim,
                               NPAR=n_params, RL0=0.0, RL1=15.0, NMX=8000, DSMAX=0.05,
                               name='c2.2:k_pa', bidirectional=True)

# limit cycle continuation
# c2_b5_lc_sols, c2_b5_lc_cont = a.run(starting_point='HB1', origin=c2_b5_cont, c='qif2b', ICP=[2, 11],
#                                      NDIM=n_dim, NPAR=n_params, RL0=-5.0, RL1=10.0, NMX=4000, DSMAX=0.1,
#                                      name='c2.2:eta_p:lc1')
# c2_b5_lc2_sols, c2_b5_lc2_cont = a.run(starting_point='HB2', origin=c2_b5_cont, c='qif2b', ICP=[2, 11],
#                                        NDIM=n_dim, NPAR=n_params, RL0=-5.0, RL1=10.0, NMX=4000, DSMAX=0.1,
#                                        name='c2.2:eta_p:lc2')

# continuation of hopf curve
c2_b5_2d1_sols, c2_b5_2d1_cont = a.run(starting_point='HB1', origin=c2_b5_cont, c='qif2', ICP=[22, 2], NDIM=n_dim,
                                       NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.1,
                                       name='c2.2:k_gp/eta_p:hb1', bidirectional=True)
c2_b5_2d2_sols, c2_b5_2d2_cont = a.run(starting_point='HB2', origin=c2_b5_cont, c='qif2', ICP=[22, 2], NDIM=n_dim,
                                       NPAR=n_params, RL0=0.0, RL1=10.0, NMX=6000, DSMAX=0.1,
                                       name='c2.2:k_gp/eta_p:hb2', bidirectional=True)

# save results
a.to_file(fname, **kwargs)
