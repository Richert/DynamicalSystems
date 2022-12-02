from pyrates.utility.pyauto import PyAuto, get_from_solutions, fractal_dimension, continue_period_doubling_bf
import numpy as np
import matplotlib.pyplot as plt

#####################################
# bifurcation analysis of qif model #
#####################################

# config
n_dim = 22
n_params = 24
a = PyAuto("auto_files", auto_dir='~/PycharmProjects/auto-07p')
c1 = False
c2 = True

store_params = ['PAR(20)', 'PAR(21)', 'PAR(11)', 'PAR(14)']
store_vars = ['U(2)', 'U(4)', 'U(21)', 'U(22)']

###################################
# condition 1: oscillatory regime #
###################################

if c1:

    # induce oscillations
    #####################

    # step 1: continuation of k_pp
    s0_sols, s0_cont = a.run(e='gpe_2pop_forced', c='qif_lc', ICP=[6, 11], NPAR=n_params, name='k_pp:1',
                             NDIM=n_dim, RL0=0.0, RL1=5.1, NMX=6000, DSMAX=0.1, STOP=['UZ1'], UZR={6: 5.0})

    # step 2: continuation of eta_p
    s1_sols, s1_cont = a.run(starting_point='UZ1', c='qif_lc', ICP=[2, 11], NPAR=n_params, name='eta_p:1',
                             NDIM=n_dim, RL0=0.1, RL1=50.0, origin=s0_cont, NMX=6000, DSMAX=0.1,
                             UZR={2: [40.0]}, STOP={'UZ1'}, variables=store_vars, params=store_params)

    starting_point = 'UZ1'
    starting_cont = s1_cont

    # continuation of driver
    ########################

    # driver parameter boundaries
    alpha_min = 0.0
    alpha_max = 2.0
    omega_min = 50.0
    omega_max = 80.0

    # step 1: codim 1 investigation of excitatory driver strength
    c0_sols, c0_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif_lc', ICP=[21, 11],
                             NPAR=n_params, name='c1:alpha', NDIM=n_dim, NMX=2000, DSMAX=0.05, RL0=alpha_min,
                             RL1=alpha_max, STOP={}, UZR={21: [1.45]}, variables=store_vars, params=store_params)

    # step 2: codim 1 investigation of of inhibitory driver strength
    c1_sols, c1_cont = a.run(starting_point='UZ1', origin=c0_cont, c='qif_lc', ICP=[23, 11], NPAR=n_params,
                             name='c1:omega', NDIM=n_dim, NMX=8000, DSMAX=0.05, RL0=omega_min, RL1=omega_max, STOP={},
                             UZR={23: [63.0, 64.0]}, bidirectional=True, variables=store_vars, params=store_params)

    # step 3: codim 2 investigation of torus bifurcations found in step 1 and 2
    i, j = 0, 0
    for s in c1_sols.v1():
        if 'TR' in s['bifurcation']:
            i += 1
            p_tmp = f'TR{i}'
            c2_sols, c2_cont = a.run(starting_point=p_tmp, origin=c1_cont, c='qif3', ICP=[23, 21, 11],
                                     NPAR=n_params, name=f'c1:omega/alpha/{p_tmp}', NDIM=n_dim, NMX=2000,
                                     DSMAX=0.05, RL0=omega_min, RL1=omega_max, STOP={'BP1'}, UZR={},
                                     bidirectional=True, variables=store_vars, params=store_params)
        elif 'PD' in s['bifurcation']:
            j += 1
            p_tmp = f'PD{j}'
            c3_sols, c3_cont = a.run(starting_point=p_tmp, origin=c1_cont, c='qif3', ICP=[23, 21, 11],
                                     NPAR=n_params, name=f'c1:omega/alpha/{p_tmp}', NDIM=n_dim, NMX=2000,
                                     DSMAX=0.05, RL0=omega_min, RL1=omega_max, STOP={'BP1', 'R22'}, UZR={},
                                     bidirectional=True, variables=store_vars, params=store_params)

        # save results
        fname = '../results/gpe_2pop_forced_lc.pkl'
        kwargs = {}
        a.to_file(fname, **kwargs)

    # step 4: codim 1 investigation of of inhibitory driver strength for omega = 63.0
    c4_sols, c4_cont = a.run(starting_point='UZ1', origin=c1_cont, c='qif_lc', ICP=[21, 11], NPAR=n_params,
                             name='c1:alpha:2', NDIM=n_dim, NMX=8000, DSMAX=0.05, RL0=alpha_min, RL1=alpha_max,
                             STOP={}, UZR={}, bidirectional=True, variables=store_vars, params=store_params)

    i += 1
    a.run(starting_point=f'TR1', origin=c4_cont, c='qif3', ICP=[23, 21, 11],
          NPAR=n_params, name=f'c1:omega/alpha/TR{i}', NDIM=n_dim, NMX=2000,
          DSMAX=0.05, RL0=omega_min, RL1=omega_max, STOP={'BP1'}, UZR={},
          bidirectional=True, variables=store_vars, params=store_params)

    c5_sols, c5_cont = a.run(starting_point='UZ2', origin=c1_cont, c='qif_lc', ICP=[21, 11], NPAR=n_params,
                             name='c1:alpha:3', NDIM=n_dim, NMX=8000, DSMAX=0.05, RL0=alpha_min, RL1=alpha_max,
                             STOP={}, UZR={}, bidirectional=True, variables=store_vars, params=store_params)

    i += 1
    a.run(starting_point=f'LP1', origin=c5_cont, c='qif3', ICP=[23, 21, 11],
          NPAR=n_params, name=f'c1:omega/alpha/TR{i}', NDIM=n_dim, NMX=2000,
          DSMAX=0.05, RL0=omega_min, RL1=omega_max, STOP={'BP1'}, UZR={},
          bidirectional=True, variables=store_vars, params=store_params)

    # save results
    fname = '../results/gpe_2pop_forced_lc.pkl'
    kwargs = {}
    a.to_file(fname, **kwargs)

##################################
# condition 2: no GPe-a feedback #
##################################

if c2:

    # induce oscillations
    #####################

    # step 1: continuation of k_pp
    s0_sols, s0_cont = a.run(e='gpe_2pop_forced', c='qif_lc', ICP=[6, 11], NPAR=n_params, name='k_pp:1',
                             NDIM=n_dim, RL0=0.0, RL1=5.1, NMX=6000, DSMAX=0.1, STOP=['UZ1'], UZR={6: 5.0})

    # step 2: continuation of eta_p
    s1_sols, s1_cont = a.run(starting_point='UZ1', c='qif_lc', ICP=[2, 11], NPAR=n_params, name='eta_p:1',
                             NDIM=n_dim, RL0=0.1, RL1=50.0, origin=s0_cont, NMX=6000, DSMAX=0.1,
                             UZR={2: [40.0]}, STOP={'UZ1'}, variables=store_vars, params=store_params)

    # step 3: continuation of k_pa
    s2_sols, s2_cont = a.run(starting_point='UZ1', c='qif_lc', ICP=[8, 11], NPAR=n_params, name='k_pa:1',
                             NDIM=n_dim, RL0=-0.1, origin=s1_cont, NMX=6000, DSMAX=0.1, DS='-',
                             UZR={8: [0.0]}, STOP={'UZ1'}, variables=store_vars, params=store_params)

    starting_point = 'UZ1'
    starting_cont = s2_cont

    # continuation of driver
    ########################

    # driver parameter boundaries
    alpha_min = 0.0
    alpha_max = 2.0
    omega_min = 50.0
    omega_max = 80.0

    # step 1: codim 1 investigation of excitatory driver strength
    c0_sols, c0_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif_lc', ICP=[21, 11],
                             NPAR=n_params, name='c1:alpha', NDIM=n_dim, NMX=2000, DSMAX=0.05, RL0=alpha_min,
                             RL1=alpha_max, STOP={}, UZR={21: [1.45]}, variables=store_vars, params=store_params)

    # step 2: codim 1 investigation of of inhibitory driver strength
    c1_sols, c1_cont = a.run(starting_point='UZ1', origin=c0_cont, c='qif_lc', ICP=[23, 11], NPAR=n_params,
                             name='c1:omega', NDIM=n_dim, NMX=8000, DSMAX=0.05, RL0=omega_min, RL1=omega_max, STOP={},
                             UZR={23: [63.0, 64.0]}, bidirectional=True, variables=store_vars, params=store_params)

    # step 3: codim 2 investigation of torus bifurcations found in step 1 and 2
    i, j = 0, 0
    for s in c1_sols.v1():
        if 'TR' in s['bifurcation']:
            i += 1
            p_tmp = f'TR{i}'
            c2_sols, c2_cont = a.run(starting_point=p_tmp, origin=c1_cont, c='qif3', ICP=[23, 21, 11],
                                     NPAR=n_params, name=f'c1:omega/alpha/{p_tmp}', NDIM=n_dim, NMX=2000,
                                     DSMAX=0.05, RL0=omega_min, RL1=omega_max, STOP={'BP1'}, UZR={},
                                     bidirectional=True, variables=store_vars, params=store_params)
        elif 'PD' in s['bifurcation']:
            j += 1
            p_tmp = f'PD{j}'
            c3_sols, c3_cont = a.run(starting_point=p_tmp, origin=c1_cont, c='qif3', ICP=[23, 21, 11],
                                     NPAR=n_params, name=f'c1:omega/alpha/{p_tmp}', NDIM=n_dim, NMX=2000,
                                     DSMAX=0.05, RL0=omega_min, RL1=omega_max, STOP={'BP1', 'R22'}, UZR={},
                                     bidirectional=True, variables=store_vars, params=store_params)

        # save results
        fname = '../results/gpe_2pop_forced_lc2.pkl'
        kwargs = {}
        a.to_file(fname, **kwargs)

    # # step 4: codim 1 investigation of of inhibitory driver strength for omega = 63.0
    # c4_sols, c4_cont = a.run(starting_point='UZ1', origin=c1_cont, c='qif_lc', ICP=[21, 11], NPAR=n_params,
    #                          name='c1:alpha:2', NDIM=n_dim, NMX=8000, DSMAX=0.05, RL0=alpha_min, RL1=alpha_max,
    #                          STOP={}, UZR={}, bidirectional=True, variables=store_vars, params=store_params)
    #
    # i += 1
    # a.run(starting_point=f'TR1', origin=c4_cont, c='qif3', ICP=[23, 21, 11],
    #       NPAR=n_params, name=f'c1:omega/alpha/TR{i}', NDIM=n_dim, NMX=2000,
    #       DSMAX=0.05, RL0=omega_min, RL1=omega_max, STOP={'BP1'}, UZR={},
    #       bidirectional=True, variables=store_vars, params=store_params)
    #
    # c5_sols, c5_cont = a.run(starting_point='UZ2', origin=c1_cont, c='qif_lc', ICP=[21, 11], NPAR=n_params,
    #                          name='c1:alpha:3', NDIM=n_dim, NMX=8000, DSMAX=0.05, RL0=alpha_min, RL1=alpha_max,
    #                          STOP={}, UZR={}, bidirectional=True, variables=store_vars, params=store_params)
    #
    # i += 1
    # a.run(starting_point=f'LP1', origin=c5_cont, c='qif3', ICP=[23, 21, 11],
    #       NPAR=n_params, name=f'c1:omega/alpha/TR{i}', NDIM=n_dim, NMX=2000,
    #       DSMAX=0.05, RL0=omega_min, RL1=omega_max, STOP={'BP1'}, UZR={},
    #       bidirectional=True, variables=store_vars, params=store_params)

    # save results
    fname = '../results/gpe_2pop_forced_lc2.pkl'
    kwargs = {}
    a.to_file(fname, **kwargs)
