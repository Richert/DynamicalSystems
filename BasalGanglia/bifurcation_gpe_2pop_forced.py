from pyrates.utility.pyauto import PyAuto
import numpy as np
import matplotlib.pyplot as plt

#####################################
# bifurcation analysis of qif model #
#####################################

# config
n_dim = 20
n_params = 25
a = PyAuto("auto_files")
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

    alphas = np.arange(35.0, 42.0, 0.5)
    omegas = np.arange(72.0, 72.5, 0.05)
    n, m = len(omegas), len(alphas)

    # step 1: codim 1 investigation of driver strength
    c0_sols, c0_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif_lc', ICP=[23, 11],
                             NPAR=n_params, name='c1:alpha', NDIM=n_dim, NMX=2000, DSMAX=0.05, RL0=0.0, RL1=42.0,
                             STOP={}, UZR={23: alphas})

    # step 2: codim 1 investigation of driver period
    c1_sols, c1_cont = a.run(starting_point='TR1', origin=c0_cont, c='qif_lc', ICP=[25, 11],
                             NPAR=n_params, name='c1:omega', NDIM=n_dim, NMX=4000, DSMAX=0.05, RL0=35.0, RL1=95.0,
                             STOP={}, UZR={}, bidirectional=True)

    # step 4: codim 2 investigation of torus bifurcations found in step 1 and 2
    i, j = 0, 0
    for s in c1_sols.values():
        if 'TR' in s['bifurcation'] or 'PD' in s['bifurcation']:
            if 'TR' in s['bifurcation']:
                i += 1
                p_tmp = f'TR{i}'
            else:
                j += 1
                p_tmp = f'PD{j}'
            c2_sols, c2_cont = a.run(starting_point=p_tmp, origin=c1_cont, c='qif3', ICP=[23, 25, 11],
                                     NPAR=n_params, name=f'c1:alpha/omega/{p_tmp}', NDIM=n_dim, NMX=3000, DSMAX=0.01,
                                     RL0=0.0, RL1=45.0, STOP={'BP1', 'R25'}, UZR={}, bidirectional=True)
            m, n = 0, 0
            for s2 in c2_sols.values():
                if 'R3' in s['bifurcation'] or 'R4' in s['bifurcation']:
                    if 'R3' in s['bifurcation']:
                        m += 1
                        p2_tmp = f'R3{m}'
                    else:
                        n += 1
                        p2_tmp = f'R4{n}'
                    c3_sols, c3_cont = a.run(starting_point=p2_tmp, origin=c2_cont, c='qif3', ICP=[23, 25, 11],
                                             NPAR=n_params, name=f'c1:alpha/omega/{p2_tmp}', NDIM=n_dim, NMX=3000,
                                             DSMAX=0.01, RL0=0.0, RL1=45.0, STOP={'BP1', 'R25'}, UZR={},
                                             bidirectional=True)

        # save results
        fname = '../results/gpe_2pop_forced_lc_all.pkl'
        kwargs = {'alpha': alphas, 'omega': omegas}
        a.to_file(fname, **kwargs)

    # # save results
    # fname = '../results/gpe_2pop_forced_lc2.pkl'
    # kwargs = {'alpha': alphas, 'omega': omegas}
    # a.to_file(fname, **kwargs)

    # # step 3: perform 1-d continuations in omega at each point in alpha continuation and extract LEs at each user point
    # LE_max = np.zeros((n, m))
    # D_ky = np.zeros_like(LE_max)
    # i = 1
    # for s in c0_sols.values():
    #     if 'UZ' in s['bifurcation']:
    #         s_tmp, _ = a.run(starting_point=f'UZ{i}', c='qif_lc', ICP=[25, 11], UZR={25: omegas}, STOP={},
    #                          get_lyapunov_exp=True, DSMAX=0.05, RL0=70.0, RL1=78.0, origin=c0_cont, NMX=2000,
    #                          bidirectional=True, NDIM=n_dim, NPAR=n_params)
    #         i += 1
    #         for s2 in s_tmp.values():
    #             if 'UZ' in s2['bifurcation']:
    #                 idx_c = np.argmin(np.abs(s2['PAR(23)'] - alphas))
    #                 idx_r = np.argmin(np.abs(s2['PAR(25)'] - omegas))
    #                 lyapunovs = s2['lyapunov_exponents']
    #                 LE_max[idx_r, idx_c] = np.max(lyapunovs)
    #                 D_ky[idx_r, idx_c] = fractal_dimension(lyapunov_exponents=lyapunovs)
    #
    # # save results
    # fname = '../results/gpe_2pop_forced_lc2.pkl'
    # kwargs = {'alpha': alphas, 'omega': omegas, 'LE_max': LE_max, 'D_ky': D_ky}
    # a.to_file(fname, **kwargs)

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
    fname = '../results/gpe_2pop_forced_bs.pkl'
    kwargs = {'alpha': alphas}
    a.to_file(fname, **kwargs)
