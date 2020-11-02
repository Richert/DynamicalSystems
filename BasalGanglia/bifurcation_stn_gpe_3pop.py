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

c1 = False   # bistable
c2 = True   # oscillatory

#########################
# initial continuations #
#########################

# continuation in time
t_sols, t_cont = a.run(e='stn_gpe_3pop', c='ivp', ICP=14, NMX=1000000, name='t', UZR={14: 1000.0}, STOP={'UZ1'},
                       NDIM=n_dim, NPAR=n_params)
starting_point = 'UZ1'
starting_cont = t_cont

# choose relative strength of inter- vs. intra-population coupling inside GPe
ki_sols, ki_cont = a.run(starting_point=starting_point, c='qif', ICP=24, NPAR=n_params, name='k_i', NDIM=n_dim,
                         RL0=0.8, RL1=2.0, origin=starting_cont, NMX=2000, DSMAX=0.1, bidirectional=True,
                         UZR={24: [0.9, 1.8]}, STOP={})

################################################################################
# c2: investigation of STN-GPe behavior near the oscillatory regime of the GPe #
################################################################################

if c2:

    fname = '../results/stn_gpe_osc_c2.pkl'
    starting_point = 'UZ1'
    starting_cont = ki_cont
    cond = "c2"

    # preparation of healthy state
    ##############################

    # continuation of eta_p
    c2_b1_sols, c2_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params,
                                   name=f'{cond}:eta_p:tmp', NDIM=n_dim, RL0=-10.0, RL1=20.0, origin=starting_cont,
                                   NMX=4000, DSMAX=0.1, UZR={2: [6.0]}, bidirectional=True)
    starting_point = 'UZ1'
    starting_cont = c2_b1_cont

    # continuation of eta_a
    c2_b2_sols, c2_b2_cont = a.run(starting_point=starting_point, c='qif', ICP=3, NPAR=n_params,
                                   name=f'{cond}:eta_a:tmp', NDIM=n_dim, RL0=-10.0, RL1=10.0, origin=starting_cont,
                                   NMX=4000, DSMAX=0.1, UZR={3: [-2.0]}, bidirectional=True)
    starting_point = 'UZ1'
    starting_cont = c2_b2_cont

    # continuation of k_ae
    c2_b3_sols, c2_b3_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif', ICP=6, NDIM=n_dim,
                                   NPAR=n_params, RL1=10.0, NMX=4000, DSMAX=0.05, name=f'{cond}:k_ae:tmp',
                                   UZR={6: [4.0]})

    # continuaiton of k_gp
    c2_b4_sols, c2_b4_cont = a.run(starting_point='UZ1', origin=c2_b3_cont, c='qif', ICP=22, NDIM=n_dim,
                                   NPAR=n_params, RL0=0.0, RL1=20.0, NMX=8000, DSMAX=0.1, name=f'{cond}:k_gp',
                                   UZR={22: [3.5, 7.0, 14.0]})

    # STN control over GPe behavior in healthy state
    ################################################

    # continuation of k_pe
    c2_b5_sols, c2_b5_cont = a.run(starting_point='UZ1', origin=c2_b4_cont, c='qif', ICP=5, NDIM=n_dim,
                                   NPAR=n_params, RL0=0.0, RL1=20.0, NMX=12000, DSMAX=0.1, name=f'{cond}:k_pe',
                                   bidirectional=True, UZR={5: [5.0]})

    # continuation of k_ae
    _, c2_b6_cont = a.run(starting_point='UZ1', origin=c2_b5_cont, c='qif', ICP=6, NDIM=n_dim,
                          NPAR=n_params, RL0=0.0, RL1=30.0, NMX=16000, DSMAX=0.1, name=f'{cond}:k_ae',
                          bidirectional=True)

    # 2D continuation of k_pe and k_ae
    c2_b5_2d1_sols, c2_b5_2d1_cont = a.run(starting_point='HB1', origin=c2_b5_cont, c='qif2', ICP=[5, 6], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=20.0, NMX=20000, DSMAX=0.1,
                                           name=f'{cond}:k_pe/k_ae:hb1', bidirectional=True)
    c2_b6_2d1_sols, c2_b6_2d1_cont = a.run(starting_point='LP1', origin=c2_b6_cont, c='qif2', ICP=[5, 6], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=20.0, NMX=20000, DSMAX=0.1,
                                           name=f'{cond}:k_pe/k_ae:lp1', bidirectional=True)
    c2_b6_2d2_sols, c2_b6_2d2_cont = a.run(starting_point='HB1', origin=c2_b6_cont, c='qif2', ICP=[5, 6], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=20.0, NMX=20000, DSMAX=0.1,
                                           name=f'{cond}:k_pe/k_ae:hb2', bidirectional=True)

    # STN control over GPe behavior in early PD state
    #################################################

    # continuation of k_pe
    c2_b7_sols, c2_b7_cont = a.run(starting_point='UZ2', origin=c2_b4_cont, c='qif', ICP=5, NDIM=n_dim,
                                   NPAR=n_params, RL0=0.0, RL1=20.0, NMX=12000, DSMAX=0.1, name=f'{cond}.2:k_pe',
                                   bidirectional=True, UZR={5: [6.0]})

    # continuation of k_ae
    _, c2_b8_cont = a.run(starting_point='UZ1', origin=c2_b7_cont, c='qif', ICP=6, NDIM=n_dim,
                          NPAR=n_params, RL0=0.0, RL1=30.0, NMX=30000, DSMAX=0.1, name=f'{cond}.2:k_ae',
                          bidirectional=True)

    # 2D continuation of k_pe and k_ae
    c2_b7_2d1_sols, c2_b7_2d1_cont = a.run(starting_point='LP1', origin=c2_b7_cont, c='qif2', ICP=[5, 6], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=20.0, NMX=10000, DSMAX=0.1,
                                           name=f'{cond}.2:k_pe/k_ae:lp1', bidirectional=True)
    c2_b7_2d2_sols, c2_b7_2d2_cont = a.run(starting_point='HB1', origin=c2_b7_cont, c='qif2', ICP=[5, 6], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=20.0, NMX=10000, DSMAX=0.1,
                                           name=f'{cond}.2:k_pe/k_ae:hb1', bidirectional=True)
    c2_b7_2d3_sols, c2_b7_2d3_cont = a.run(starting_point='HB2', origin=c2_b7_cont, c='qif2', ICP=[5, 6], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=20.0, NMX=10000, DSMAX=0.1,
                                           name=f'{cond}.2:k_pe/k_ae:hb2', bidirectional=True)
    c2_b7_2d4_sols, c2_b7_2d4_cont = a.run(starting_point='HB3', origin=c2_b7_cont, c='qif2', ICP=[5, 6], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=20.0, NMX=10000, DSMAX=0.1,
                                           name=f'{cond}.2:k_pe/k_ae:hb3', bidirectional=True)
    c2_b8_2d1_sols, c2_b8_2d1_cont = a.run(starting_point='HB3', origin=c2_b8_cont, c='qif2', ICP=[5, 6], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=20.0, NMX=24000, DSMAX=0.1,
                                           name=f'{cond}.2:k_pe/k_ae:hb4', bidirectional=True)
    c2_b8_2d2_sols, c2_b8_2d2_cont = a.run(starting_point='HB1', origin=c2_b8_cont, c='qif2', ICP=[5, 6], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=20.0, NMX=24000, DSMAX=0.1,
                                           name=f'{cond}.2:k_pe/k_ae:hb5', bidirectional=True)

    # STN control over GPe behavior in advanced PD state
    ####################################################

    # continuation of k_pe
    c2_b9_sols, c2_b9_cont = a.run(starting_point='UZ3', origin=c2_b4_cont, c='qif', ICP=5, NDIM=n_dim,
                                   NPAR=n_params, RL0=0.0, RL1=20.0, NMX=12000, DSMAX=0.1, name=f'{cond}.3:k_pe',
                                   bidirectional=True, UZR={5: [11.0]})

    # continuation of k_ae
    _, c2_b10_cont = a.run(starting_point='UZ1', origin=c2_b9_cont, c='qif', ICP=6, NDIM=n_dim,
                           NPAR=n_params, RL0=0.0, RL1=30.0, NMX=20000, DSMAX=0.1, name=f'{cond}.3:k_ae',
                           bidirectional=True)

    # 2D continuation of k_pe and k_ae
    c2_b9_2d1_sols, c2_b9_2d1_cont = a.run(starting_point='HB1', origin=c2_b9_cont, c='qif2', ICP=[5, 6], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=20.0, NMX=10000, DSMAX=0.1,
                                           name=f'{cond}.3:k_pe/k_ae:hb1', bidirectional=True)
    c2_b9_2d2_sols, c2_b9_2d2_cont = a.run(starting_point='HB2', origin=c2_b9_cont, c='qif2', ICP=[5, 6], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=20.0, NMX=10000, DSMAX=0.1,
                                           name=f'{cond}.3:k_pe/k_ae:hb2', bidirectional=True)
    c2_b9_2d3_sols, c2_b9_2d3_cont = a.run(starting_point='HB3', origin=c2_b9_cont, c='qif2', ICP=[5, 6], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=20.0, NMX=30000, DSMAX=0.1,
                                           name=f'{cond}.3:k_pe/k_ae:hb3', bidirectional=True)
    c2_b9_2d4_sols, c2_b9_2d4_cont = a.run(starting_point='HB4', origin=c2_b9_cont, c='qif2', ICP=[5, 6], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=20.0, NMX=10000, DSMAX=0.1,
                                           name=f'{cond}.3:k_pe/k_ae:hb4', bidirectional=True)
    c2_b10_2d1_sols, c2_b10_2d1_cont = a.run(starting_point='LP1', origin=c2_b10_cont, c='qif2', ICP=[5, 6], NDIM=n_dim,
                                             NPAR=n_params, RL0=0.0, RL1=20.0, NMX=10000, DSMAX=0.1,
                                             name=f'{cond}.3:k_pe/k_ae:lp1', bidirectional=True)
    c2_b10_2d2_sols, c2_b10_2d2_cont = a.run(starting_point='HB3', origin=c2_b10_cont, c='qif2', ICP=[5, 6], NDIM=n_dim,
                                             NPAR=n_params, RL0=0.0, RL1=20.0, NMX=10000, DSMAX=0.1,
                                             name=f'{cond}.3:k_pe/k_ae:hb5', bidirectional=True)

elif c1:

    fname = '../results/stn_gpe_osc_c1.pkl'
    starting_point = 'UZ2'
    starting_cont = ki_cont
    cond = "c1"

    # preparation of healthy state
    ##############################

    # continuation of eta_p
    c1_b1_sols, c1_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params,
                                   name=f'{cond}:eta_p:tmp', NDIM=n_dim, RL0=-10.0, RL1=20.0, origin=starting_cont,
                                   NMX=4000, DSMAX=0.1, UZR={2: [3.2]}, bidirectional=True)
    starting_point = 'UZ1'
    starting_cont = c1_b1_cont

    # continuation of eta_a
    c1_b2_sols, c1_b2_cont = a.run(starting_point=starting_point, c='qif', ICP=3, NPAR=n_params,
                                   name=f'{cond}:eta_a:tmp', NDIM=n_dim, RL0=-10.0, RL1=10.0, origin=starting_cont,
                                   NMX=4000, DSMAX=0.1, UZR={3: [2.0]}, bidirectional=True)
    starting_point = 'UZ1'
    starting_cont = c1_b2_cont

    # continuation of k_ae
    c1_b3_sols, c1_b3_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif', ICP=6, NDIM=n_dim,
                                   NPAR=n_params, RL1=10.0, NMX=4000, DSMAX=0.05, name=f'{cond}:k_ae:tmp',
                                   UZR={6: [4.0]})

    # continuaiton of k_gp
    c1_b4_sols, c1_b4_cont = a.run(starting_point='UZ1', origin=c1_b3_cont, c='qif', ICP=22, NDIM=n_dim,
                                   NPAR=n_params, RL0=0.0, RL1=10.0, NMX=8000, DSMAX=0.1, name=f'{cond}:k_gp',
                                   UZR={22: [3.0, 6.0, 9.0]})

    # STN control over GPe behavior in healthy state
    ################################################

    # continuation of k_pe
    c1_b5_sols, c1_b5_cont = a.run(starting_point='UZ1', origin=c1_b4_cont, c='qif', ICP=5, NDIM=n_dim,
                                   NPAR=n_params, RL0=0.0, RL1=30.0, NMX=16000, DSMAX=0.1, name=f'{cond}:k_pe',
                                   bidirectional=True, UZR={5: [10.0]})

    # continuation of k_ae
    # _, c1_b6_cont = a.run(starting_point='UZ1', origin=c1_b5_cont, c='qif', ICP=6, NDIM=n_dim,
    #                       NPAR=n_params, RL0=0.0, RL1=30.0, NMX=24000, DSMAX=0.1, name=f'{cond}:k_ae',
    #                       bidirectional=True, UZR={6: [25.0]})
    #
    # # second continuation of k_pe
    # _, c1_b5_cont2 = a.run(starting_point='UZ1', origin=c1_b6_cont, c='qif', ICP=5, NDIM=n_dim,
    #                        NPAR=n_params, RL0=0.0, RL1=60.0, NMX=20000, DSMAX=0.1, name=f'{cond}:k_pe:2',
    #                        bidirectional=True)

    # 2D continuation of k_pe and k_ae
    c1_b5_2d1_sols, c1_b5_2d1_cont = a.run(starting_point='LP1', origin=c1_b5_cont, c='qif2', ICP=[5, 6], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=30.0, NMX=10000, DSMAX=0.1,
                                           name=f'{cond}:k_pe/k_ae:lp1', bidirectional=True)
    c1_b5_2d2_sols, c1_b5_2d2_cont = a.run(starting_point='HB1', origin=c1_b5_cont, c='qif2', ICP=[5, 6], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=30.0, NMX=20000, DSMAX=0.1,
                                           name=f'{cond}:k_pe/k_ae:hb1', bidirectional=True)
    c1_b5_2d3_sols, c1_b5_2d3_cont = a.run(starting_point='HB3', origin=c1_b5_cont, c='qif2', ICP=[5, 6], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=30.0, NMX=10000, DSMAX=0.1,
                                           name=f'{cond}:k_pe/k_ae:hb2', bidirectional=True)

    # STN control over GPe behavior in early PD state
    #################################################

    # continuation of k_pe
    c1_b7_sols, c1_b7_cont = a.run(starting_point='UZ2', origin=c1_b4_cont, c='qif', ICP=5, NDIM=n_dim,
                                   NPAR=n_params, RL0=0.0, RL1=30.0, NMX=16000, DSMAX=0.1, name=f'{cond}.2:k_pe',
                                   bidirectional=True, UZR={5: [15.0]})

    # continuation of k_ae
    # _, c1_b8_cont = a.run(starting_point='UZ1', origin=c1_b7_cont, c='qif', ICP=6, NDIM=n_dim,
    #                       NPAR=n_params, RL0=0.0, RL1=30.0, NMX=24000, DSMAX=0.1, name=f'{cond}.2:k_ae',
    #                       bidirectional=True, UZR={6: [5.0]})
    #
    # # second continuation of k_pe
    # _, c1_b7_cont2 = a.run(starting_point='UZ1', origin=c1_b8_cont, c='qif', ICP=5, NDIM=n_dim,
    #                        NPAR=n_params, RL0=0.0, RL1=60.0, NMX=20000, DSMAX=0.1, name=f'{cond}.2:k_pe:2',
    #                        bidirectional=True)

    # 2D continuation of k_pe and k_ae
    c1_b7_2d1_sols, c1_b7_2d1_cont = a.run(starting_point='LP1', origin=c1_b7_cont, c='qif2', ICP=[5, 6], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=30.0, NMX=10000, DSMAX=0.1,
                                           name=f'{cond}.2:k_pe/k_ae:lp1', bidirectional=True)
    c1_b7_2d2_sols, c1_b7_2d2_cont = a.run(starting_point='HB1', origin=c1_b7_cont, c='qif2', ICP=[5, 6], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=30.0, NMX=10000, DSMAX=0.1,
                                           name=f'{cond}.2:k_pe/k_ae:hb1', bidirectional=True)
    c1_b7_2d3_sols, c1_b7_2d3_cont = a.run(starting_point='HB2', origin=c1_b7_cont, c='qif2', ICP=[5, 6], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=30.0, NMX=20000, DSMAX=0.1,
                                           name=f'{cond}.2:k_pe/k_ae:hb2', bidirectional=True)
    c1_b7_2d4_sols, c1_b7_2d4_cont = a.run(starting_point='HB4', origin=c1_b7_cont, c='qif2', ICP=[5, 6], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=30.0, NMX=10000, DSMAX=0.1,
                                           name=f'{cond}.2:k_pe/k_ae:hb3', bidirectional=True)

    # STN control over GPe behavior in advanced PD state
    ####################################################

    # continuation of k_pe
    c1_b9_sols, c1_b9_cont = a.run(starting_point='UZ3', origin=c1_b4_cont, c='qif', ICP=5, NDIM=n_dim,
                                   NPAR=n_params, RL0=0.0, RL1=30.0, NMX=16000, DSMAX=0.1, name=f'{cond}.3:k_pe',
                                   bidirectional=True, UZR={5: [15.0]})

    # continuation of k_ae
    # _, c1_b10_cont = a.run(starting_point='UZ1', origin=c1_b9_cont, c='qif', ICP=6, NDIM=n_dim,
    #                        NPAR=n_params, RL0=0.0, RL1=30.0, NMX=24000, DSMAX=0.4, name=f'{cond}.3:k_ae',
    #                        bidirectional=True, UZR={6: [25.0]})
    #
    # # second continuation of k_pe
    # _, c1_b9_cont2 = a.run(starting_point='UZ1', origin=c1_b10_cont, c='qif', ICP=5, NDIM=n_dim,
    #                        NPAR=n_params, RL0=0.0, RL1=60.0, NMX=20000, DSMAX=0.1, name=f'{cond}.3:k_pe:2',
    #                        bidirectional=True)

    # 2D continuation of k_pe and k_ae
    c1_b9_2d1_sols, c1_b9_2d1_cont = a.run(starting_point='LP1', origin=c1_b9_cont, c='qif2', ICP=[5, 6], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=30.0, NMX=10000, DSMAX=0.1,
                                           name=f'{cond}.3:k_pe/k_ae:lp1', bidirectional=True)
    c1_b9_2d2_sols, c1_b9_2d2_cont = a.run(starting_point='HB1', origin=c1_b9_cont, c='qif2', ICP=[5, 6], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=30.0, NMX=10000, DSMAX=0.1,
                                           name=f'{cond}.3:k_pe/k_ae:hb1', bidirectional=True)
    c1_b9_2d3_sols, c1_b9_2d3_cont = a.run(starting_point='HB3', origin=c1_b9_cont, c='qif2', ICP=[5, 6], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=30.0, NMX=20000, DSMAX=0.1,
                                           name=f'{cond}.3:k_pe/k_ae:hb2', bidirectional=True)
    c1_b9_2d4_sols, c1_b9_2d4_cont = a.run(starting_point='HB4', origin=c1_b9_cont, c='qif2', ICP=[5, 6], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=30.0, NMX=10000, DSMAX=0.1,
                                           name=f'{cond}.3:k_pe/k_ae:hb3', bidirectional=True)
    c1_b9_2d5_sols, c1_b9_2d5_cont = a.run(starting_point='HB5', origin=c1_b9_cont, c='qif2', ICP=[5, 6], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=30.0, NMX=10000, DSMAX=0.1,
                                           name=f'{cond}.3:k_pe/k_ae:hb4', bidirectional=True)

# save results
a.to_file(fname, **kwargs)
