from pyrates.utility.pyauto import PyAuto, get_from_solutions, fractal_dimension
import numpy as np
import matplotlib.pyplot as plt

#####################################
# bifurcation analysis of qif model #
#####################################

# config
n_dim = 20
n_params = 25
a = PyAuto("auto_files")
c1 = [False, True]
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

if any(c1):

    starting_point = 'UZ4'
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
                             NMX=6000, DSMAX=0.1, UZR={2: [4.0]}, STOP={'UZ1'})

    starting_point = 'UZ1'
    starting_cont = s3_cont

    # continuation of eta_a
    s4_sols, s4_cont = a.run(starting_point=starting_point, c='qif_lc', ICP=[3, 11], NPAR=n_params,
                             name='c1:eta_a', NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=starting_cont,
                             NMX=6000, DSMAX=0.1, UZR={3: [-5.0]}, STOP={'UZ1'}, DS='-')

    starting_point = 'UZ1'
    starting_cont = s4_cont

    if c1[0]:

        # continuation of driver
        ########################

        # driver parameter boundaries
        alpha_min = 0.0
        alpha_max = 100.0
        omega_min = 25.0
        omega_max = 100.0

        # step 1: codim 1 investigation of driver strength
        c0_sols, c0_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif_lc', ICP=[23, 11],
                                 NPAR=n_params, name='c1:alpha', NDIM=n_dim, NMX=2000, DSMAX=0.05, RL0=alpha_min,
                                 RL1=alpha_max, STOP={}, UZR={23: [35.0]})

        # step 2: codim 1 investigation of driver period
        c1_sols, c1_cont = a.run(starting_point='UZ1', origin=c0_cont, c='qif_lc', ICP=[25, 11],
                                 NPAR=n_params, name='c1:omega', NDIM=n_dim, NMX=8000, DSMAX=0.05, RL0=omega_min,
                                 RL1=omega_max, STOP={}, UZR={}, bidirectional=True)

        # step 3: codim 2 investigation of torus bifurcations found in step 1 and 2
        i, j = 0, 0
        for s in c1_sols.values():
            if 'TR' in s['bifurcation']:
                i += 1
                p_tmp = f'TR{i}'
                c2_sols, c2_cont = a.run(starting_point=p_tmp, origin=c1_cont, c='qif3', ICP=[25, 23, 11],
                                         NPAR=n_params, name=f'c1:omega/alpha/{p_tmp}', NDIM=n_dim, NMX=2000,
                                         DSMAX=0.05, RL0=omega_min, RL1=omega_max, STOP={'BP1', 'R21', 'R11'}, UZR={},
                                         bidirectional=True)
                bfs = get_from_solutions(['bifurcation'], c2_sols)
                if "R2" in bfs:
                    s_tmp, c_tmp = a.run(starting_point='R21', origin=c2_cont, c='qif_lc', ICP=[25, 11],
                                         NPAR=n_params, name='c1:omega/R21', NDIM=n_dim, NMX=1000, DSMAX=0.01,
                                         RL0=omega_min, RL1=omega_max, STOP={'PD1', 'TR1'}, UZR={})
                    pds = get_from_solutions(['bifurcation'], s_tmp)
                    if "PD" in pds:
                        j += 1
                        p2_tmp = f'PD{j}'
                        c2_sols, c2_cont = a.run(starting_point='PD1', origin=c_tmp, c='qif3', ICP=[25, 23, 11],
                                                 NPAR=n_params, name=f'c1:omega/alpha/{p2_tmp}', NDIM=n_dim, NMX=2000,
                                                 DSMAX=0.05, RL0=omega_min, RL1=omega_max, STOP={'BP1', 'R22'}, UZR={},
                                                 bidirectional=True)

            # save results
            fname = '../results/gpe_2pop_forced_lc.pkl'
            kwargs = {}
            a.to_file(fname, **kwargs)

    if c1[1]:

        # lyapunov/dimension mapping
        ############################

        # driver parameter boundaries
        alpha_min = 0.0
        alpha_max = 60.0
        omega_min = 25.0
        omega_max = 100.0

        # driver parameter grid
        n = 100
        alphas = np.round(np.linspace(40.0, 50.0, num=n), decimals=3)
        omegas = np.round(np.linspace(52.0, 62.0, num=n), decimals=3)

        # step 1: codim 1 investigation of driver strength
        c0_sols, c0_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif_lc', ICP=[23, 11],
                                 NPAR=n_params, name='c1:alpha', NDIM=n_dim, NMX=2000, DSMAX=0.05, RL0=alpha_min,
                                 RL1=alpha_max, STOP={}, UZR={23: alphas})

        # step 2: codim 1 investigation of driver period with lyapunov exponent/fractal dimension extraction
        alpha_col = []
        omega_col = []
        le_max_col = []
        fd_col = []
        i = 1
        for point in c0_sols.values():
            if 'UZ' in point['bifurcation']:
                c1_sols, c1_cont = a.run(starting_point=f'UZ{i}', origin=c0_cont, c='qif_lc', ICP=[25, 11],
                                         NPAR=n_params, name='c1:omega', NDIM=n_dim, NMX=8000, DSMAX=0.05,
                                         RL0=omega_min, RL1=omega_max, STOP={}, UZR={25: omegas}, bidirectional=True,
                                         get_lyapunov_exp=True)
                for data in get_from_solutions(['bifurcation', 'PAR(23)', 'PAR(25)', 'lyapunov_exponents'], c1_sols):
                    if 'UZ' in data[0]:
                        alpha_col.append(data[1])
                        omega_col.append(data[2])
                        le_max_col.append(np.max(data[3]))
                        fd_col.append(fractal_dimension(data[3]))
                i += 1

        # save results
        fname = '../results/gpe_2pop_forced_lc_chaos.pkl'
        kwargs = {'alphas': alpha_col, 'omegas': omega_col, 'lyapunovs': le_max_col, 'fractal_dimensions': fd_col}
        a.to_file(fname, **kwargs)

################################
# condition 2: bistable regime #
################################

if c2:

    starting_point = 'UZ4'
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
                             NMX=1000, DSMAX=0.1, UZR={2: [2.53]}, STOP={'UZ1'})

    starting_point = 'UZ1'
    starting_cont = s3_cont

    # continuation of driver
    ########################

    # step 1: codim 1 investigation of driver strength
    c0_sols, c0_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif_lc', ICP=[23, 11],
                             NPAR=n_params, name='c2:alpha', NDIM=n_dim, NMX=8000, DSMAX=0.05, RL0=0.0, RL1=50.0,
                             STOP={}, UZR={}, NPR=10)

    # step 2: codim 2 investigation of torus bifurcation found in step 1
    c1_sols, c1_cont = a.run(starting_point='LP1', origin=c0_cont, c='qif3', ICP=[25, 23, 11],
                             NPAR=n_params, name='c2:alpha/omega', NDIM=n_dim, NMX=8000, DSMAX=0.1, RL0=10.0, RL1=100.0,
                             STOP={}, UZR={}, bidirectional=True, NPR=10)

    # save results
    fname = '../results/gpe_2pop_forced_bs.pkl'
    kwargs = {}
    a.to_file(fname, **kwargs)
