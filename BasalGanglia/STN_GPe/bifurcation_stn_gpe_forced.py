from pyrates.utility.pyauto import PyAuto, get_from_solutions, fractal_dimension, continue_period_doubling_bf
import numpy as np
import matplotlib.pyplot as plt

#####################################
# bifurcation analysis of qif model #
#####################################

# config
n_dim = 39
n_params = 31
a = PyAuto("auto_files", auto_dir='~/PycharmProjects/auto-07p')
c1 = [False, False, True]
c2 = False

store_params = ['PAR(23)', 'PAR(25)', 'PAR(14)']
store_vars = ['U(1)', 'U(3)', 'U(5)', 'U(38)']

# initial continuations
#######################

# continuation of total intrinsic coupling strengths inside GPe
s0_sols, s0_cont = a.run(e='gpe_2pop_forced', c='qif_lc', ICP=[19, 11], NPAR=n_params, name='k_gp',
                         NDIM=n_dim, RL0=0.0, RL1=100.0, NMX=6000, DSMAX=0.5, UZR={19: [10.0, 15.0, 20.0, 25.0, 30.0]},
                         STOP={'UZ5'}, variables=store_vars, params=store_params)

# continuation of GPe-p projection strength
s1_sols, s1_cont = a.run(starting_point='UZ5', ICP=[20, 11], NPAR=n_params, name='k_p',
                         NDIM=n_dim, RL0=0.0, RL1=100.0, NMX=6000, DSMAX=0.5, UZR={20: [1.5]},
                         STOP={'UZ1'}, variables=store_vars, params=store_params, origin=s0_cont)

###################################
# condition 1: oscillatory regime #
###################################

if any(c1):

    starting_point = 'UZ1'
    starting_cont = s1_cont

    # continuation of between vs. within population coupling strengths
    s2_sols, s2_cont = a.run(starting_point=starting_point, c='qif_lc', ICP=[21, 11], NPAR=n_params, name='c1:k_i',
                             NDIM=n_dim, RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.1, DS='-',
                             UZR={21: [0.9]}, STOP={'UZ1'}, variables=store_vars, params=store_params)

    starting_point = 'UZ1'
    starting_cont = s2_cont

    # continuation of eta_p
    s3_sols, s3_cont = a.run(starting_point=starting_point, c='qif_lc', ICP=[2, 11], NPAR=n_params,
                             name='c1:eta_p', NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=starting_cont,
                             NMX=6000, DSMAX=0.1, UZR={2: [4.8]}, STOP={'UZ1'}, variables=store_vars,
                             params=store_params)

    starting_point = 'UZ1'
    starting_cont = s3_cont

    # continuation of eta_a
    s4_sols, s4_cont = a.run(starting_point=starting_point, c='qif_lc', ICP=[3, 11], NPAR=n_params,
                             name='c1:eta_a', NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=starting_cont,
                             NMX=6000, DSMAX=0.1, UZR={3: [-6.5]}, STOP={'UZ1'}, DS='-', variables=store_vars,
                             params=store_params)

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
                                 RL1=alpha_max, STOP={}, UZR={23: [30.0]}, variables=store_vars, params=store_params)

        # step 2: codim 1 investigation of driver period
        c1_sols, c1_cont = a.run(starting_point='UZ1', origin=c0_cont, c='qif_lc', ICP=[25, 11],
                                 NPAR=n_params, name='c1:omega', NDIM=n_dim, NMX=8000, DSMAX=0.05, RL0=omega_min,
                                 RL1=omega_max, STOP={}, UZR={25: [77.3]}, bidirectional=True, variables=store_vars,
                                 params=store_params)

        # step 3: codim 2 investigation of torus bifurcations found in step 1 and 2
        i, j = 0, 0
        for s in c1_sols.v1():
            if 'TR' in s['bifurcation']:
                i += 1
                p_tmp = f'TR{i}'
                c2_sols, c2_cont = a.run(starting_point=p_tmp, origin=c1_cont, c='qif3', ICP=[25, 23, 11],
                                         NPAR=n_params, name=f'c1:omega/alpha/{p_tmp}', NDIM=n_dim, NMX=2000,
                                         DSMAX=0.05, RL0=omega_min, RL1=omega_max, STOP={'BP1', 'R21', 'R11'}, UZR={},
                                         bidirectional=True, variables=store_vars, params=store_params)
                bfs = get_from_solutions(['bifurcation'], c2_sols)
                if "R2" in bfs:
                    s_tmp, c_tmp = a.run(starting_point='R21', origin=c2_cont, c='qif_lc', ICP=[25, 11],
                                         NPAR=n_params, name='c1:omega/R21', NDIM=n_dim, NMX=500, DSMAX=0.001,
                                         RL0=omega_min, RL1=omega_max, STOP={'PD1', 'TR1'}, UZR={},
                                         variables=store_vars, params=store_params, DS='-')
                    pds = get_from_solutions(['bifurcation'], s_tmp)
                    if "PD" in pds:
                        j += 1
                        p2_tmp = f'PD{j}'
                        c2_sols, c2_cont = a.run(starting_point='PD1', origin=c_tmp, c='qif3', ICP=[25, 23, 11],
                                                 NPAR=n_params, name=f'c1:omega/alpha/{p2_tmp}', NDIM=n_dim, NMX=2000,
                                                 DSMAX=0.05, RL0=omega_min, RL1=omega_max, STOP={'BP1', 'R22'}, UZR={},
                                                 bidirectional=True, variables=store_vars, params=store_params)

                # save results
                fname = '../results/gpe_2pop_forced_lc.pkl'
                kwargs = {}
                a.to_file(fname, **kwargs)

        # step 4: continue the period doubling bifurcations in driver strength that we found above
        c3_sols, c3_cont = a.run(starting_point='UZ1', origin=c1_cont, c='qif_lc', ICP=[23, 11],
                                 NPAR=n_params, name='c1:alpha2', NDIM=n_dim, NMX=8000, DSMAX=0.05, RL0=alpha_min,
                                 RL1=alpha_max, STOP={}, bidirectional=True, variables=store_vars,
                                 params=store_params)
        pds, a = continue_period_doubling_bf(solution=c3_sols, continuation=c3_cont, pyauto_instance=a,
                                             c='qif2b', ICP=[23, 11], NMX=2500, DSMAX=0.05, NTST=800, ILP=0, NDIM=n_dim,
                                             get_timeseries=True, get_lyapunov_exp=True, NPR=10, STOP={'BP1'},
                                             RL0=alpha_min, RL1=alpha_max)
        pds.append('c1:alpha2')

        # save results
        fname = '../results/gpe_2pop_forced_lc.pkl'
        kwargs = {'pd_solutions': pds}
        a.to_file(fname, **kwargs)

    if c1[1]:

        # lyapunov/dimension mapping
        ############################

        # driver parameter boundaries
        alpha_min = 0.0
        alpha_max = 100.0
        omega_min = 25.0
        omega_max = 100.0

        # driver parameter grid
        n = 100
        alphas = np.round(np.linspace(70.0, 90.0, num=n), decimals=3)
        omegas = np.round(np.linspace(60.0, 70.0, num=n), decimals=3)

        # step 1: codim 1 investigation of driver strength
        c0_sols, c0_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif_lc', ICP=[23, 11],
                                 NPAR=n_params, name='c1:alpha', NDIM=n_dim, NMX=2000, DSMAX=0.05, RL0=alpha_min,
                                 RL1=alpha_max, STOP={}, UZR={23: alphas}, variables=store_vars, params=store_params)

        # step 2: codim 1 investigation of driver period with lyapunov exponent/fractal dimension extraction
        alpha_col = []
        omega_col = []
        le_max_col = []
        fd_col = []
        i = 1
        for point in c0_sols.v1():
            if 'UZ' in point['bifurcation']:
                c1_sols, c1_cont = a.run(starting_point=f'UZ{i}', origin=c0_cont, c='qif_lc', ICP=[25, 11],
                                         NPAR=n_params, name='c1:omega', NDIM=n_dim, NMX=8000, DSMAX=0.05,
                                         RL0=omega_min, RL1=omega_max, STOP={}, UZR={25: omegas}, bidirectional=True,
                                         get_lyapunov_exp=True, variables=store_vars, params=store_params)
                for data in get_from_solutions(['bifurcation', 'PAR(23)', 'PAR(25)', 'lyapunov_exponents'], c1_sols):
                    if 'UZ' in data[0]:
                        alpha_col.append(data[1])
                        omega_col.append(data[2])
                        if len(data[3]) > 0:
                            le_max_col.append(np.max(data[3]))
                            fd_col.append(fractal_dimension(data[3]))
                        else:
                            le_max_col.append(0.0)
                            fd_col.append(0.0)
                i += 1

        # save results
        fname = '../results/gpe_2pop_forced_lc_chaos.pkl'
        kwargs = {'alphas': alpha_col, 'omegas': omega_col, 'lyapunovs': le_max_col, 'fractal_dimensions': fd_col}
        a.to_file(fname, **kwargs)

    if c1[2]:

        # continuation of k_ap
        s5_sols, s5_cont = a.run(starting_point=starting_point, c='qif_lc', ICP=[7, 11], NPAR=n_params,
                                 name='c1:k_pa', NDIM=n_dim, RL0=-0.01, RL1=2.0, origin=starting_cont,
                                 NMX=6000, DSMAX=0.1, UZR={7: [0.0]}, STOP={'UZ1'}, DS='-', variables=store_vars,
                                 params=store_params)

        starting_point = 'UZ1'
        starting_cont = s5_cont

        # continuation of k_pp
        s6_sols, s6_cont = a.run(starting_point=starting_point, c='qif_lc', ICP=[9, 11], NPAR=n_params,
                                 name='c1:k_aa', NDIM=n_dim, RL0=0.0, RL1=10.0, origin=starting_cont,
                                 NMX=6000, DSMAX=0.1, UZR={9: [5.0]}, variables=store_vars,
                                 params=store_params, STOP=['UZ1'])

        starting_point = 'UZ1'
        starting_cont = s6_cont

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
                                 RL1=alpha_max, STOP={}, UZR={23: [30.0]}, variables=store_vars, params=store_params)

        # step 2: codim 1 investigation of driver period
        c1_sols, c1_cont = a.run(starting_point='UZ1', origin=c0_cont, c='qif_lc', ICP=[25, 11],
                                 NPAR=n_params, name='c1:omega', NDIM=n_dim, NMX=8000, DSMAX=0.05, RL0=omega_min,
                                 RL1=omega_max, STOP={}, UZR={25: [77.3]}, bidirectional=True, variables=store_vars,
                                 params=store_params)

        # step 3: codim 2 investigation of torus bifurcations found in step 1 and 2
        i, j = 0, 0
        for s in c1_sols.v1():
            if 'TR' in s['bifurcation']:
                i += 1
                p_tmp = f'TR{i}'
                c2_sols, c2_cont = a.run(starting_point=p_tmp, origin=c1_cont, c='qif3', ICP=[25, 23, 11],
                                         NPAR=n_params, name=f'c1:omega/alpha/{p_tmp}', NDIM=n_dim, NMX=2000,
                                         DSMAX=0.05, RL0=omega_min, RL1=omega_max, STOP={'BP1', 'R21', 'R11'}, UZR={},
                                         bidirectional=True, variables=store_vars, params=store_params)
                bfs = get_from_solutions(['bifurcation'], c2_sols)
                if "R2" in bfs:
                    s_tmp, c_tmp = a.run(starting_point='R21', origin=c2_cont, c='qif_lc', ICP=[25, 11],
                                         NPAR=n_params, name='c1:omega/R21', NDIM=n_dim, NMX=500, DSMAX=0.001,
                                         RL0=omega_min, RL1=omega_max, STOP={'PD1', 'TR1'}, UZR={},
                                         variables=store_vars, params=store_params, DS='-')
                    pds = get_from_solutions(['bifurcation'], s_tmp)
                    if "PD" in pds:
                        j += 1
                        p2_tmp = f'PD{j}'
                        c2_sols, c2_cont = a.run(starting_point='PD1', origin=c_tmp, c='qif3', ICP=[25, 23, 11],
                                                 NPAR=n_params, name=f'c1:omega/alpha/{p2_tmp}', NDIM=n_dim, NMX=2000,
                                                 DSMAX=0.05, RL0=omega_min, RL1=omega_max, STOP={'BP1', 'R22'}, UZR={},
                                                 bidirectional=True, variables=store_vars, params=store_params)

                # save results
                fname = '../results/gpe_2pop_forced_lc2.pkl'
                kwargs = {}
                a.to_file(fname, **kwargs)

        # step 4: continue the period doubling bifurcations in driver strength that we found above
        c3_sols, c3_cont = a.run(starting_point='UZ1', origin=c1_cont, c='qif_lc', ICP=[23, 11],
                                 NPAR=n_params, name='c1:alpha2', NDIM=n_dim, NMX=8000, DSMAX=0.05, RL0=alpha_min,
                                 RL1=alpha_max, STOP={}, bidirectional=True, variables=store_vars,
                                 params=store_params)
        pds, a = continue_period_doubling_bf(solution=c3_sols, continuation=c3_cont, pyauto_instance=a,
                                             c='qif2b', ICP=[23, 11], NMX=2500, DSMAX=0.05, NTST=800, ILP=0, NDIM=n_dim,
                                             get_timeseries=True, get_lyapunov_exp=True, NPR=10, STOP={'BP1'},
                                             RL0=alpha_min, RL1=alpha_max)
        pds.append('c1:alpha2')

        # save results
        fname = '../results/gpe_2pop_forced_lc2.pkl'
        kwargs = {'pd_solutions': pds}
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
                             bidirectional=True, UZR={21: [0.5, 0.75, 1.5, 1.8]}, STOP={'UZ2'})

    starting_point = 'UZ4'
    starting_cont = s2_cont

    # continuation of eta_p
    s3_sols, s3_cont = a.run(starting_point=starting_point, c='qif_lc', ICP=[2, 11], NPAR=n_params,
                             name='c2:eta_p', NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=starting_cont,
                             NMX=1000, DSMAX=0.1, UZR={2: [3.2]}, STOP={'UZ1'})

    starting_point = 'UZ1'
    starting_cont = s3_cont

    # continuation of eta_a
    s4_sols, s4_cont = a.run(starting_point=starting_point, c='qif_lc', ICP=[3, 11], NPAR=n_params,
                             name='c1:eta_a', NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=starting_cont,
                             NMX=6000, DSMAX=0.1, UZR={3: [3.0]}, STOP={'UZ1'}, variables=store_vars,
                             params=store_params)

    starting_point = 'UZ1'
    starting_cont = s4_cont

    # continuation of driver
    ########################

    # driver parameter boundaries
    alpha_min = 0.0
    alpha_max = 100.0
    omega_min = 25.0
    omega_max = 100.0

    # step 1: codim 1 investigation of driver strength
    c2_sols, c2_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif_lc', ICP=[23, 11],
                             NPAR=n_params, name='c2:alpha', NDIM=n_dim, NMX=2000, DSMAX=0.05, RL0=alpha_min,
                             RL1=alpha_max, STOP={}, UZR={23: [30.0]}, variables=store_vars, params=store_params)

    # step 2: codim 1 investigation of driver period
    c1_sols, c1_cont = a.run(starting_point='UZ1', origin=c2_cont, c='qif_lc', ICP=[25, 11],
                             NPAR=n_params, name='c1:omega', NDIM=n_dim, NMX=8000, DSMAX=0.05, RL0=omega_min,
                             RL1=omega_max, STOP={}, UZR={25: [77.3]}, bidirectional=True, variables=store_vars,
                             params=store_params)

    # save results
    fname = '../results/gpe_2pop_forced_bs.pkl'
    kwargs = {}
    a.to_file(fname, **kwargs)
