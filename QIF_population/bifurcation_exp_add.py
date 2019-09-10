from pyauto import PyAuto
import matplotlib.pyplot as plt
import numpy as np

#########################################
# configs, descriptions and definitions #
#########################################

# problem description
#####################

"""
Performs continuation of extended Montbrio population given the initial condition:

 U1: r_0 = 0.114741, 
 U2: v_0 = -2.774150,
 U3: e_0 = 0.0,

with parameters:

 PAR1: eta = -10.0
 PAR2: J = 15.0
 PAR3: alpha = 0.0
 PAR4: tau = 1.0
 PAR5: D = 2.0

"""


# configuration
###############

# configuration
codim1 = True
codim2 = True
n_grid_points = 100
n_dim = 3
n_params = 6
eta_cont_idx = 3

# definitions
#############


def continue_period_doubling_bf(solution, continuation, auto_runner, max_iter=100, iteration=0):
    solutions = []
    pd_idx = 0
    for point, point_info in solution.items():
        if 'PD' in point_info['bifurcation']:
            s_tmp, _ = auto_runner.run(starting_point=point, c='qif2b', ICP=[1, 11], NMX=2000, NTST=600, DSMAX=0.05,
                                       ILP=0, NDIM=n_dim, origin=continuation, get_timeseries=True,
                                       get_lyapunov_exp=True, name=f'pd_{pd_idx}')
            solutions.append(f'pd_{pd_idx}')
            if iteration >= max_iter:
                break
            else:
                solutions += continue_period_doubling_bf(s_tmp, continuation, auto_runner, iteration=iteration + 1)
    return solutions


###################################
# parameter continuations in auto #
###################################

a = PyAuto("auto_files")

# initial continuation in the adaptation strength alpha
alpha_0 = [0.1, 0.2, 0.4, 0.8, 1.6]
alpha_solutions, alpha_cont = a.run(e='qif_exp_add', c='qif', ICP=3, UZR={3: alpha_0}, NDIM=n_dim,
                                    STOP=['UZ' + str(len(alpha_0))], DSMAX=0.005, NMX=4000, name='s0')

# principle continuation in eta
###############################

# continue in eta for each adaptation rate alpha
solutions_eta = list()
solutions_eta.append(a.run(starting_point='EP1', ICP=1, DSMAX=0.005, RL1=0.0, RL0=-12.0, NMX=2000, origin=alpha_cont,
                           bidirectional=True, name='eta_0', NDIM=n_dim))

i = 1
for point, point_info in alpha_solutions.items():
    if 'UZ' in point_info['bifurcation']:
        solutions_eta.append(a.run(starting_point=point, ICP=1, DSMAX=0.005, RL1=0.0, RL0=-12.0, NMX=4000,
                                   origin=alpha_cont, bidirectional=True, name=f"eta_{i}", NDIM=n_dim))
        i += 1

# choose a continuation in eta to run further continuations on
eta_points, eta_cont = solutions_eta[eta_cont_idx]

if codim1:

    # limit cycle continuation of hopf bifurcations in eta
    eta_hb1_solutions, eta_hb1_cont = a.run(starting_point='HB1', c='qif2b', ICP=[1, 11], DSMAX=0.05, NMX=3000,
                                            origin=eta_cont, name='eta_hb1', NDIM=n_dim)
    eta_hb2_solutions, eta_hb2_cont = a.run(starting_point='HB2', c='qif2b', ICP=[1, 11], DSMAX=0.05, NMX=3000,
                                            origin=eta_cont, name='eta_hb2', NDIM=n_dim, get_lyapunov_exp=True,
                                            get_timeseries=True)

    # continue the period doubling bifurcations in eta that we found above
    pds = continue_period_doubling_bf(solution=eta_hb2_solutions, continuation=eta_hb2_cont, auto_runner=a)
    pds.append('eta_hb2')

    # continuation in eta and alpha
    ###############################

    if codim2:

        # continue the limit cycle borders in alpha and eta
        eta_alpha_hb2_solutions, eta_alpha_hb2_cont = a.run(starting_point='HB2', c='qif2', ICP=[1, 3], DSMAX=0.01,
                                                            NMX=4000, bidirectional=True, origin=eta_cont,
                                                            name='eta_alpha_hb2', NDIM=n_dim)

        # continue the first period doubling of the limit cycle
        eta_alpha_pd_solutions, eta_alpha_pd_cont = a.run(starting_point='PD1', c='qif3', ICP=[1, 3],
                                                          NMX=3500, DSMAX=0.05, origin=eta_hb2_cont,
                                                          bidirectional=True, name='eta_alpha_pd', NTST=600,
                                                          NDIM=n_dim)

################
# save results #
################

fname = '../results/exp_add.hdf5'

kwargs = {}
if codim1:
    kwargs['pd_solutions'] = pds
a.to_file(fname, **kwargs)
