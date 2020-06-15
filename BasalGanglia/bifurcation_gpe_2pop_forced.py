from pyauto import PyAuto
import numpy as np
import matplotlib.pyplot as plt

#####################################
# bifurcation analysis of qif model #
#####################################

# config
n_dim = 20
n_params = 25
a = PyAuto("auto_files")
fname = '../results/gpe_2pop_forced.pkl'
c1 = True
c2 = False

# initial continuations
#######################

# continuation of total intrinsic coupling strengths inside GPe
s0_sols, s0_cont = a.run(e='gpe_2pop_forced', c='qif_lc', ICP=[19, 11], NPAR=n_params, name='k_gp',
                         NDIM=n_dim, RL0=0.0, RL1=100.0, NMX=6000, DSMAX=0.5, UZR={19: [10.0, 15.0, 20.0, 25.0]},
                         STOP={'UZ4'})

###################################
# condition 1: oscillatory regime #
###################################

if c1:

    starting_point = 'UZ3'
    starting_cont = s0_cont

    # continuation of between vs. within population coupling strengths
    s2_sols, s2_cont = a.run(starting_point=starting_point, c='qif_lc', ICP=[21, 11], NPAR=n_params, name='c1:k_i',
                             NDIM=n_dim, RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.1,
                             bidirectional=True, UZR={21: [0.5, 0.75, 1.5, 2.0]}, STOP={'UZ2'})

    starting_point = 'UZ1'
    starting_cont = s2_cont

    # continuation of eta_p
    s3_sols, s3_cont = a.run(starting_point=starting_point, c='qif_lc', ICP=[2, 11], NPAR=n_params,
                             name='c1:eta_p', NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=starting_cont,
                             NMX=6000, DSMAX=0.1, UZR={2: [3.0]}, STOP={'UZ1'})

    starting_point = 'UZ1'
    starting_cont = s3_cont

    # continuation of eta_a
    s4_sols, s4_cont = a.run(starting_point=starting_point, c='qif_lc', ICP=[3, 11], NPAR=n_params,
                             name='c1:eta_a', NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=starting_cont,
                             NMX=6000, DSMAX=0.1, UZR={3: [-5.0]}, STOP={'UZ1'}, DS='-')

    starting_point = 'UZ1'
    starting_cont = s4_cont

    # continuation of driver
    ########################

    alphas = np.arange(10.0, 100.0, 10.0)

    # step 1: codim 1 investigation of driver strength
    c0_sols, c0_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif_lc', ICP=[23, 11],
                             NPAR=n_params, name='c1:alpha', NDIM=n_dim, NMX=4000, DSMAX=0.05, RL0=0.0, RL1=100.0,
                             STOP={}, UZR={23: alphas})

    # step 2: codim 2 investigation of torus bifurcation found in step 1
    c1_sols, c1_cont = a.run(starting_point='TR1', origin=c0_cont, c='qif3', ICP=[23, 25, 11],
                             NPAR=n_params, name='c1:alpha/omega', NDIM=n_dim, NMX=2000, DSMAX=0.5, RL0=0.0, RL1=100.0,
                             STOP={'UZ1'}, UZR={25: [50.0, 100.0]})
    c2_sols, c2_cont = a.run(starting_point='EP1', origin=c1_cont, bidirectional=True)

    # save results
    kwargs = {'alpha': alphas}
    a.to_file(fname, **kwargs)

################################
# condition 2: bistable regime #
################################

if c2:

    starting_point = 'UZ3'
    starting_cont = s0_cont

    # continuation of between vs. within population coupling strengths
    s2_sols, s2_cont = a.run(starting_point=starting_point, c='qif_lc', ICP=[21, 11], NPAR=n_params, name='c2:k_i',
                             NDIM=n_dim, RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.1,
                             bidirectional=True, UZR={21: [0.5, 0.75, 1.5, 2.0]}, STOP={'UZ2'})

    starting_point = 'UZ4'
    starting_cont = s2_cont

    # continuation of eta_p
    s3_sols, s3_cont = a.run(starting_point=starting_point, c='qif_lc', ICP=[2, 11], NPAR=n_params,
                             name='c2:eta_p', NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=starting_cont,
                             NMX=6000, DSMAX=0.1, UZR={2: [2.0]}, STOP={'UZ3'})

    starting_point = 'UZ3'
    starting_cont = s3_cont

    # continuation of driver
    ########################

    alphas = np.arange(10.0, 100.0, 10.0)

    # step 1: codim 1 investigation of driver strength
    c0_sols, c0_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif_lc', ICP=[23, 11],
                             NPAR=n_params, name='c2:alpha', NDIM=n_dim, NMX=4000, DSMAX=0.05, RL0=0.0, RL1=100.0,
                             STOP={'LP1'}, UZR={23: alphas})

    # step 2: codim 2 investigation of torus bifurcation found in step 1
    c1_sols, c1_cont = a.run(starting_point='LP1', origin=c0_cont, c='qif3', ICP=[23, 25, 11],
                             NPAR=n_params, name='c2:alpha/omega', NDIM=n_dim, NMX=2000, DSMAX=0.5, RL0=0.0, RL1=100.0,
                             STOP={'UZ1'}, UZR={25: [50.0, 100.0]})
    c2_sols, c2_cont = a.run(starting_point='EP1', origin=c1_cont, bidirectional=True)

    # save results
    kwargs = {'alpha': alphas}
    a.to_file(fname, **kwargs)
