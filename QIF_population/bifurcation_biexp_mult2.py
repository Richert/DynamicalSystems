import sys
sys.path.append('../')
import os
from pyrates.utility.pyauto import PyAuto
import numpy as np

#########################################
# configs, descriptions and definitions #
#########################################

# problem description
"""
Performs continuation of extended Montbrio population given the initial condition:

 U1: r_0 = 0.114741, 
 U2: v_0 = -2.774150,
 U3: e_0 = 0.0,
 U4: a_0 = 0.0

with initial parameters:

 PAR1: eta = -10.0
 PAR2: J = 15.0
 PAR3: alpha = 0.0
 PAR4: tau = 1.0
 PAR5: D = 2.0

"""

# configuration
codim1 = True
codim2 = False
period_mapping = False
n_grid_points = 100
n_dim = 4
n_params = 6

###################################
# parameter continuations in auto #
###################################

a = PyAuto("auto_files")

# initial continuation in the adaptation strength alpha
alpha_0 = [0.0125, 0.025, 0.05, 0.1, 0.2]
alpha_solutions, alpha_cont = a.run(e='qif_biexp_mult', c='qif', ICP=3, UZR={3: alpha_0},
                                    STOP=['UZ' + str(len(alpha_0))], DSMAX=0.005, NMX=4000, name='s0',
                                    NDIM=n_dim, NPAR=n_params)

tau_points, tau_cont = a.run(starting_point='UZ3', ICP=4, DSMAX=0.001, RL1=10.01, RL0=0.001, NMX=10000,
                             origin=alpha_cont, bidirectional=True, name='tau_0', NDIM=n_dim, NPAR=n_params,
                             UZR={4: 0.3})

eta_points, eta_cont = a.run(starting_point='UZ1', ICP=1, DSMAX=0.001, RL1=0.0, RL0=-12.0, NMX=100000, origin=tau_cont,
                             bidirectional=True, name='eta_0', NDIM=n_dim, NPAR=n_params)

if codim1:

    # 1D continuations in eta and tau1
    ##################################

    eta_hb1_solutions, eta_hb1_cont = a.run(starting_point='HB1', c='qif2b', ICP=[1, 11], DSMAX=0.01, NMX=4000,
                                            origin=eta_cont, name='eta/hb1', STOP={'BP1', 'LP10'}, NDIM=n_dim,
                                            NPAR=n_params, UZR={1: [-5.2]})
    eta_hb2_solutions, eta_hb2_cont = a.run(starting_point='HB2', c='qif2b', ICP=[1, 11], DSMAX=0.01, NMX=2000,
                                            origin=eta_cont, name='eta/hb2', NDIM=n_dim, NPAR=n_params)

    if codim2:

        # 2D continuations in tau1, eta and alpha
        #########################################

        tau_min = 0.01
        tau_max = 11.0

        # continue the limit cycle borders in tau_r and tau_d
        eta_tau2_hb2_solutions, eta_tau2_hb2_cont = a.run(starting_point='HB2', c='qif2', ICP=[4, 1], DSMAX=0.01,
                                                          NMX=4000, bidirectional=True, origin=eta_cont, RL0=tau_min,
                                                          RL1=tau_max, name='tau_r/eta/hb2', ILP=0,
                                                          NDIM=n_dim, NPAR=n_params)

        # continue the generalized hopf in tau_r
        tau_r_gh_solutions, tau_r_gh_cont = a.run(starting_point='GH1', c='qif2b', ICP=[4, 1, 11], DSMAX=0.005,
                                                  RL0=tau_min, NDIM=n_dim, NPAR=n_params, RL1=tau_max, NMX=1000,
                                                  origin=eta_tau2_hb2_cont, name='tau_r/eta/gh1')

        # continue the period doubling region of the limit cycle
        eta_tau2_pd_solutions, eta_tau2_pd_cont = a.run(starting_point='PD1', c='qif3', ICP=[4, 1, 11],
                                                        NMX=4000, DSMAX=0.01, STOP=[], NDIM=n_dim, NPAR=n_params,
                                                        origin=tau_r_gh_cont, bidirectional=True, RL0=tau_min,
                                                        RL1=tau_max, name='tau_r/eta/lc_pd1',
                                                        )

        # # continue the period doubling region of the limit cycle
        # eta_tau2_lp_solutions, eta_tau2_lp_cont = a.run(starting_point='LP1', c='qif3', ICP=[4, 5, 11],
        #                                                 NMX=4000, DSMAX=0.01, STOP=[], NDIM=n_dim, NPAR=n_params,
        #                                                 origin=eta_tau2_pd_cont, bidirectional=True, RL0=tau_min,
        #                                                 RL1=tau_max, name='tau_r/tau_d/lc_lp1',
        #                                                 )

        # continue the limit cycle borders in alpha and tau1
        tau1_alpha_hb2_solution, tau1_alpha_hb2_cont = a.run(starting_point='HB2', c='qif2', ICP=[4, 3],
                                                             DSMAX=0.01, NMX=4000, RL0=tau_min, RL1=tau_max,
                                                             origin=eta_cont, name='tau_r/alpha/hb2',
                                                             ILP=0, bidirectional=True,
                                                             NDIM=n_dim, NPAR=n_params)

        # continue the generalized hopf of the limit cycle
        tau1_alpha_hb2_lc_solutions, tau1_alpha_hb2_lc_cont = a.run(starting_point='GH1', c='qif2b', ICP=[4, 3, 11],
                                                                    NMX=4000, DSMAX=0.01, STOP=[], RL0=tau_min,
                                                                    origin=tau1_alpha_hb2_cont, RL1=tau_max,
                                                                    name='tau_r/alpha/gh1', NDIM=n_dim, NPAR=n_params)

        # continue the period doubling region of the limit cycle
        tau1_alpha_pd_solutions, tau1_alpha_pd_cont = a.run(starting_point='PD1', c='qif3', ICP=[4, 3, 11],
                                                            NMX=4000, DSMAX=0.01, STOP=[], ILP=0,
                                                            origin=tau1_alpha_hb2_lc_cont, bidirectional=True,
                                                            RL0=tau_min, RL1=tau_max, name='tau_r/alpha/lc_pd1',
                                                            NDIM=n_dim, NPAR=n_params)

        # # get limit cycle periods
        # if period_mapping:
        #
        #     etas = np.round(np.linspace(-6, -3, n_grid_points), decimals=4).tolist()
        #     period_solutions = np.zeros((len(tau2s), len(etas)))
        #     for point, point_info in eta_tau2_hb2_lc_solutions.items():
        #         if np.round(point_info['PAR(5)'], decimals=4) in tau2s:
        #             solution_tmp, cont_tmp = a.run(starting_point=point, c='qif', ICP=1, UZR={1: etas}, STOP={},
        #                                            EPSL=1e-6, EPSU=1e-6, EPSS=1e-4, ILP=0, ISP=0, IPS=2, DSMAX=0.05,
        #                                            DS='-', origin=eta_tau2_hb2_lc_cont, get_period=True)
        #             for point_tmp, point_info_tmp in solution_tmp.items():
        #                 if 'UZ' in point_info_tmp['bifurcation']:
        #                     idx_c = np.argwhere(np.round(point_info_tmp['PAR(1)'], decimals=4) == etas)
        #                     idx_r = np.argwhere(np.round(point_info_tmp['PAR(5)'], decimals=4) == tau2s)
        #                     period_solutions[idx_r, idx_c] = point_info_tmp['period']


################
# save results #
################

fname = '../results/biexp_mult2.hdf5'
a.to_file(fname)

#if period_mapping:
#    period_solutions.tofile(f"biexp_mult_period")