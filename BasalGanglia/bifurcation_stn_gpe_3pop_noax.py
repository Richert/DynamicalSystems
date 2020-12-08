from pyrates.utility.pyauto import PyAuto, get_from_solutions
import numpy as np
import sys
from matplotlib.pyplot import show

"""Bifurcation analysis of STN-GPe model with two GPe populations (arkypallidal and prototypical) and 
gamma-distributed axonal delays and bi-exponential synapses."""

path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"

# config
n_dim = 19
n_params = 25
a = PyAuto("auto_files", auto_dir=auto_dir)
kwargs = dict()
model = 'stn_gpe_noax'
c1 = False   # bistable
c2 = True   # oscillatory

#########################
# initial continuations #
#########################

# continuation in time
t_sols, t_cont = a.run(e='stn_gpe_3pop_noax', c='ivp', ICP=14, NMX=1000000, name='t', UZR={14: 1000.0}, STOP={'UZ1'},
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

    cond = "c2"
    fname = f'../results/{model}_{cond}.pkl'
    starting_point = 'UZ1'
    starting_cont = ki_cont

    # preparation of healthy state
    ##############################

    # continuation of eta_p
    c2_b1_sols, c2_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params,
                                   name=f'{cond}:eta_p:tmp', NDIM=n_dim, RL0=-10.0, RL1=20.0, origin=starting_cont,
                                   NMX=4000, DSMAX=0.1, UZR={2: [5.0]}, bidirectional=True)
    starting_point = 'UZ1'
    starting_cont = c2_b1_cont

    # continuation of eta_a
    c2_b2_sols, c2_b2_cont = a.run(starting_point=starting_point, c='qif', ICP=3, NPAR=n_params,
                                   name=f'{cond}:eta_a:tmp', NDIM=n_dim, RL0=-10.0, RL1=10.0, origin=starting_cont,
                                   NMX=4000, DSMAX=0.1, UZR={3: [-5.0]}, bidirectional=True)
    starting_point = 'UZ1'
    starting_cont = c2_b2_cont

    # continuation of k_ae
    c2_b3_sols, c2_b3_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif', ICP=6, NDIM=n_dim,
                                   NPAR=n_params, RL1=10.0, NMX=4000, DSMAX=0.05, name=f'{cond}:k_ae:tmp',
                                   UZR={6: [6.0]})

    # continuation of k_pe
    c2_b4_sols, c2_b4_cont = a.run(starting_point='UZ1', origin=c2_b3_cont, c='qif', ICP=5, NDIM=n_dim,
                                   NPAR=n_params, RL0=0.0, RL1=20.0, NMX=8000, DSMAX=0.1, name=f'{cond}:k_pe',
                                   UZR={5: [6.0]}, DS='-', STOP=['UZ1'])

    # continuation of k_gp and k_stn
    ################################

    # continuations of k_gp
    c2_b5_sols, c2_b5_cont = a.run(starting_point='UZ1', origin=c2_b4_cont, c='qif', ICP=22, NDIM=n_dim,
                                   NPAR=n_params, RL0=0.0, RL1=20.0, NMX=12000, DSMAX=0.1, name=f'{cond}:k_gp',
                                   bidirectional=True, UZR={22: [1.5, 9.0]})

    # continuations of k_stn
    c2_b6_sols, c2_b6_cont = a.run(starting_point='UZ1', origin=c2_b4_cont, c='qif', ICP=25, NDIM=n_dim,
                                   NPAR=n_params, RL0=0.0, RL1=4.0, NMX=16000, DSMAX=0.1, name=f'{cond}:k_stn',
                                   bidirectional=True)
    a.run(starting_point='HB1', origin=c2_b6_cont, c='qif2b', ICP=[25, 11], NDIM=n_dim, NPAR=n_params, RL0=0.0,
          RL1=4.0, NMX=4000, DSMAX=0.1, name=f'{cond}:k_stn:lc1', STOP=['BP1'])

    c2_b7_sols, c2_b7_cont = a.run(starting_point='UZ1', origin=c2_b5_cont, c='qif', ICP=25, NDIM=n_dim,
                                   NPAR=n_params, RL0=0.0, RL1=3.0, NMX=16000, DSMAX=0.1, name=f'{cond}:k_stn:2',
                                   bidirectional=True)
    a.run(starting_point='HB1', origin=c2_b7_cont, c='qif2b', ICP=[25, 11], NDIM=n_dim, NPAR=n_params, RL0=0.0,
          RL1=3.0, NMX=4000, DSMAX=0.1, name=f'{cond}:k_stn:2:lc1', STOP=['BP1'])

    c2_b8_sols, c2_b8_cont = a.run(starting_point='UZ2', origin=c2_b5_cont, c='qif', ICP=25, NDIM=n_dim,
                                   NPAR=n_params, RL0=0.0, RL1=5.0, NMX=12000, DSMAX=0.1, name=f'{cond}:k_stn:3',
                                   bidirectional=True)
    a.run(starting_point='HB1', origin=c2_b8_cont, c='qif2b', ICP=[25, 11], NDIM=n_dim, NPAR=n_params, RL0=0.0,
          RL1=5.0, NMX=4000, DSMAX=0.1, name=f'{cond}:k_stn:3:lc1', STOP=['BP1'])
    a.run(starting_point='HB2', origin=c2_b8_cont, c='qif2b', ICP=[25, 11], NDIM=n_dim, NPAR=n_params, RL0=0.0,
          RL1=5.0, NMX=4000, DSMAX=0.1, name=f'{cond}:k_stn:3:lc2', STOP=['BP1'])
    a.run(starting_point='HB3', origin=c2_b8_cont, c='qif2b', ICP=[25, 11], NDIM=n_dim, NPAR=n_params, RL0=0.0,
          RL1=5.0, NMX=4000, DSMAX=0.1, name=f'{cond}:k_stn:3:lc3', STOP=['BP1'])
    a.run(starting_point='HB4', origin=c2_b8_cont, c='qif2b', ICP=[25, 11], NDIM=n_dim, NPAR=n_params, RL0=0.0,
          RL1=5.0, NMX=4000, DSMAX=0.1, name=f'{cond}:k_stn:3:lc4', STOP=['BP1'])

    # 2D continuation of k_stn and k_gp
    c2_b6_2d1_sols, c2_b6_2d1_cont = a.run(starting_point='LP1', origin=c2_b6_cont, c='qif2', ICP=[25, 22], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.1, RL1=10.0, NMX=20000, DSMAX=0.1,
                                           name=f'{cond}:k_stn/k_gp:lp1', bidirectional=True)
    c2_b6_2d2_sols, c2_b6_2d2_cont = a.run(starting_point='HB1', origin=c2_b6_cont, c='qif2', ICP=[25, 22], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.1, RL1=10.0, NMX=20000, DSMAX=0.1,
                                           name=f'{cond}:k_stn/k_gp:hb1', bidirectional=True)
    c2_b7_2d1_sols, c2_b7_2d1_cont = a.run(starting_point='HB1', origin=c2_b7_cont, c='qif2', ICP=[25, 22], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.1, RL1=10.0, NMX=24000, DSMAX=0.1,
                                           name=f'{cond}:k_stn/k_gp:hb2', bidirectional=True)
    c2_b8_2d1_sols, c2_b8_2d1_cont = a.run(starting_point='HB1', origin=c2_b8_cont, c='qif2', ICP=[25, 22], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.1, RL1=10.0, NMX=24000, DSMAX=0.1,
                                           name=f'{cond}:k_stn/k_gp:hb3', bidirectional=True)
    c2_b8_2d2_sols, c2_b8_2d2_cont = a.run(starting_point='HB2', origin=c2_b8_cont, c='qif2', ICP=[25, 22], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.1, RL1=10.0, NMX=24000, DSMAX=0.1,
                                           name=f'{cond}:k_stn/k_gp:hb4', bidirectional=True)
    c2_b8_2d3_sols, c2_b8_2d3_cont = a.run(starting_point='HB3', origin=c2_b8_cont, c='qif2', ICP=[25, 22], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.1, RL1=10.0, NMX=24000, DSMAX=0.1,
                                           name=f'{cond}:k_stn/k_gp:hb5', bidirectional=True)
    c2_b8_2d4_sols, c2_b8_2d4_cont = a.run(starting_point='HB4', origin=c2_b8_cont, c='qif2', ICP=[25, 22], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.1, RL1=10.0, NMX=24000, DSMAX=0.1,
                                           name=f'{cond}:k_stn/k_gp:hb6', bidirectional=True)

elif c1:

    cond = "c1"
    fname = f'../results/{model}_{cond}.pkl'
    starting_point = 'UZ2'
    starting_cont = ki_cont

    # preparation of healthy state
    ##############################

    # continuation of eta_p
    c1_b1_sols, c1_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params,
                                   name=f'{cond}:eta_p:tmp', NDIM=n_dim, RL0=-10.0, RL1=20.0, origin=starting_cont,
                                   NMX=4000, DSMAX=0.1, UZR={2: [3.0]}, bidirectional=True)
    starting_point = 'UZ1'
    starting_cont = c1_b1_cont

    # continuation of eta_a
    c1_b2_sols, c1_b2_cont = a.run(starting_point=starting_point, c='qif', ICP=3, NPAR=n_params,
                                   name=f'{cond}:eta_a:tmp', NDIM=n_dim, RL0=-10.0, RL1=10.0, origin=starting_cont,
                                   NMX=4000, DSMAX=0.1, UZR={3: [1.0]}, bidirectional=True)
    starting_point = 'UZ1'
    starting_cont = c1_b2_cont

    # continuation of k_ae
    c1_b3_sols, c1_b3_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif', ICP=6, NDIM=n_dim,
                                   NPAR=n_params, RL1=10.0, NMX=4000, DSMAX=0.05, name=f'{cond}:k_ae:tmp',
                                   UZR={6: [6.0]})

    # continuation of k_pe
    c1_b4_sols, c1_b4_cont = a.run(starting_point='UZ1', origin=c1_b3_cont, c='qif', ICP=5, NDIM=n_dim,
                                   NPAR=n_params, RL0=0.0, RL1=20.0, NMX=8000, DSMAX=0.1, name=f'{cond}:k_pe',
                                   UZR={5: [6.0]}, DS='-', STOP=['UZ1'])

    # continuation of k_stn
    c1_b5_sols, c1_b5_cont = a.run(starting_point='UZ1', origin=c1_b4_cont, c='qif', ICP=25, NDIM=n_dim,
                                   NPAR=n_params, RL0=0.1, RL1=5.0, NMX=12000, DSMAX=0.1, name=f'{cond}:k_stn',
                                   UZR={25: [2.0]}, STOP=['UZ1'])

    # continuation of k_gp and k_stn
    ################################

    # continuations of k_gp
    c1_b6_sols, c1_b6_cont = a.run(starting_point='UZ1', origin=c1_b5_cont, c='qif', ICP=22, NDIM=n_dim,
                                   NPAR=n_params, RL0=0.0, RL1=25.0, NMX=12000, DSMAX=0.1, name=f'{cond}:k_gp',
                                   bidirectional=True, UZR={22: [5.5, 10.0, 22.0]})

    # continuation of k_stn for k_gp = 3.0
    c1_b7_sols, c1_b7_cont = a.run(starting_point='UZ1', origin=c1_b5_cont, c='qif', ICP=25, NDIM=n_dim,
                                   NPAR=n_params, RL0=0.1, RL1=5.0, NMX=12000, DSMAX=0.1, name=f'{cond}:k_stn:1',
                                   bidirectional=True, UZR={25: [4.5]})
    a.run(starting_point='HB1', origin=c1_b7_cont, c='qif2b', ICP=[25, 11], NDIM=n_dim, NPAR=n_params, RL0=0.0,
          RL1=5.0, NMX=4000, DSMAX=0.1, name=f'{cond}:k_stn:1:lc1', STOP=['BP1'])
    a.run(starting_point='HB2', origin=c1_b7_cont, c='qif2b', ICP=[25, 11], NDIM=n_dim, NPAR=n_params, RL0=0.0,
          RL1=5.0, NMX=4000, DSMAX=0.1, name=f'{cond}:k_stn:1:lc2', STOP=['BP1'])
    a.run(starting_point='HB3', origin=c1_b7_cont, c='qif2b', ICP=[25, 11], NDIM=n_dim, NPAR=n_params, RL0=0.0,
          RL1=5.0, NMX=4000, DSMAX=0.1, name=f'{cond}:k_stn:1:lc3', STOP=['BP1'])

    # continuation of k_stn for k_gp = 5.5
    c1_b8_sols, c1_b8_cont = a.run(starting_point='UZ1', origin=c1_b6_cont, c='qif', ICP=25, NDIM=n_dim,
                                   NPAR=n_params, RL0=0.1, RL1=3.0, NMX=12000, DSMAX=0.1, name=f'{cond}:k_stn:2',
                                   bidirectional=True)
    a.run(starting_point='HB1', origin=c1_b8_cont, c='qif2b', ICP=[25, 11], NDIM=n_dim, NPAR=n_params, RL0=0.0,
          RL1=3.0, NMX=4000, DSMAX=0.1, name=f'{cond}:k_stn:2:lc1', STOP=['BP1'])
    a.run(starting_point='HB2', origin=c1_b8_cont, c='qif2b', ICP=[25, 11], NDIM=n_dim, NPAR=n_params, RL0=0.0,
          RL1=3.0, NMX=6000, DSMAX=0.1, name=f'{cond}:k_stn:2:lc2', STOP=['BP1'])

    # continuation of k_stn for k_gp = 10.0
    c1_b9_sols, c1_b9_cont = a.run(starting_point='UZ2', origin=c1_b6_cont, c='qif', ICP=25, NDIM=n_dim,
                                   NPAR=n_params, RL0=0.5, RL1=5.0, NMX=12000, DSMAX=0.1, name=f'{cond}:k_stn:3',
                                   bidirectional=True)
    a.run(starting_point='HB1', origin=c1_b9_cont, c='qif2b', ICP=[25, 11], NDIM=n_dim, NPAR=n_params, RL0=0.5,
          RL1=5.0, NMX=4000, DSMAX=0.1, name=f'{cond}:k_stn:3:lc1', STOP=['BP1'])
    a.run(starting_point='HB2', origin=c1_b9_cont, c='qif2b', ICP=[25, 11], NDIM=n_dim, NPAR=n_params, RL0=0.5,
          RL1=5.0, NMX=4000, DSMAX=0.1, name=f'{cond}:k_stn:3:lc2', STOP=['BP1'])
    a.run(starting_point='HB4', origin=c1_b9_cont, c='qif2b', ICP=[25, 11], NDIM=n_dim, NPAR=n_params, RL0=0.5,
          RL1=5.0, NMX=4000, DSMAX=0.1, name=f'{cond}:k_stn:3:lc3', STOP=['BP1'])
    a.run(starting_point='HB5', origin=c1_b9_cont, c='qif2b', ICP=[25, 11], NDIM=n_dim, NPAR=n_params, RL0=0.5,
          RL1=5.0, NMX=4000, DSMAX=0.1, name=f'{cond}:k_stn:3:lc4', STOP=['BP1'])

    # continuation of k_stn for k_gp = 15.0
    c1_b10_sols, c1_b10_cont = a.run(starting_point='UZ3', origin=c1_b6_cont, c='qif', ICP=25, NDIM=n_dim,
                                     NPAR=n_params, RL0=0.0, RL1=5.0, NMX=16000, DSMAX=0.1, name=f'{cond}:k_stn:4',
                                     bidirectional=True)

    # 2D continuation of k_stn and k_gp
    c1_b7_2d1_sols, c1_b7_2d1_cont = a.run(starting_point='LP1', origin=c1_b7_cont, c='qif2', ICP=[25, 22], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.1, RL1=10.0, NMX=20000, DSMAX=0.1,
                                           name=f'{cond}:k_stn/k_gp:lp1', bidirectional=True)
    c1_b7_2d2_sols, c1_b7_2d2_cont = a.run(starting_point='HB1', origin=c1_b7_cont, c='qif2', ICP=[25, 22], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.1, RL1=10.0, NMX=20000, DSMAX=0.1,
                                           name=f'{cond}:k_stn/k_gp:hb1', bidirectional=True)
    c1_b8_2d1_sols, c1_b8_2d1_cont = a.run(starting_point='HB2', origin=c1_b8_cont, c='qif2', ICP=[25, 22], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.1, RL1=10.0, NMX=20000, DSMAX=0.1,
                                           name=f'{cond}:k_stn/k_gp:hb2', bidirectional=True)
    c1_b10_2d1_sols, c1_b10_2d1_cont = a.run(starting_point='HB2', origin=c1_b10_cont, c='qif2', ICP=[25, 22],
                                             NDIM=n_dim, NPAR=n_params, RL0=0.1, RL1=10.0, NMX=20000, DSMAX=0.1,
                                             name=f'{cond}:k_stn/k_gp:hb3', bidirectional=True)

# save results
a.to_file(fname, **kwargs)
