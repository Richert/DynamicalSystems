from pyauto import PyAuto
import numpy as np
import matplotlib.pyplot as plt

#####################################
# bifurcation analysis of qif model #
#####################################

# config
n_dim = 46
n_params = 10
a = PyAuto("auto_files")

# continuation in k
t_sols, t_cont = a.run(e='qif_stn_gpe', c='ivp', ICP=14, DS=5e-2, DSMIN=1e-4, DSMAX=1.0, NMX=1000000, name='k',
                       UZR={14: 10000.0}, STOP={'UZ1'}, NDIM=n_dim, NPAR=n_params)

# continuation in k
k_sols, k_cont = a.run(starting_point='UZ1', c='qif', ICP=1, NPAR=n_params, name='k', NDIM=n_dim,
                       RL0=-20.0, RL1=20.0, origin=t_cont)

# 2d continuation of hopf bifurcation in alpha and eta_i
k_alpha_sols, k_alpha_cont = a.run(starting_point='HB1', c='qif2', ICP=[8, 2], DSMAX=0.1, NMX=10000, NPAR=n_params,
                                   name='k_alpha', origin=k_cont, NDIM=n_dim, bidirectional=True, RL0=0.0, RL1=20.0)

# continuation of limit cycle
k_lc_sols, k_lc_cont = a.run(starting_point='HB1', c='qif2b', ICP=[3, 11], DSMAX=0.1, NMX=6000, NPAR=n_params,
                             name='k_lc', origin=k_cont, NDIM=n_dim, RL0=1.0, RL1=4.0)

# save results
fname = '../results/qif_stn_gpe.pkl'
kwargs = dict()
a.to_file(fname, **kwargs)

# if rate_adapt_syn_model:
#
#     # config
#     n_dim = 12
#     n_params = 6
#     m_ras = PyAuto("auto_files")
#
#     # continuation in eta_i
#     etai_sols, etai_cont = m_ras.run(e='stn_gpe_rate_adapt_syn', c='qif', ICP=2, DSMAX=0.05, NMX=6000, NPAR=n_params,
#                                      name='eta_i', NDIM=n_dim, STOP={}, RL0=-10.0, RL1=0.0, bidirectional=True)

# # principle continuation in eta
# ###############################
#
# # continue in alpha
# alphas = [0.005, 0.01, 0.02, 0.04, 0.08]
# n = len(alphas)
# alpha_solutions, alpha_cont = a
#
# # continue in delta
# solutions = []
# i = 0
# for point, point_info in alpha_solutions.items():
#     if 'UZ' in point_info['bifurcation']:
#         solutions.append(a.run(starting_point=f'UZ{i}', ICP=2, DSMAX=0.005, NMX=10000, NPAR=n_params,
#                                name=f'etai_{i}', NDIM=n_dim, STOP={}, origin=alpha_cont, bidirectional=True, RL0=-10.0,
#                                RL1=-1.9))
#         i += 1
#
# etai_solution, etai_cont = solutions[cont_idx]
#
# cont = solutions
# if codim1:
#
#     # limit cycle continuation of hopf bifurcations in eta_i
#     Etai_hb1_solutions, etai_hb1_cont = a.run(starting_point='HB1', ICP=[2, 11], NMX=8000, origin=etai_cont,
#                                               name='etai_hb', DSMAX=0.05, IPS=2, STOP='BP2')
#
#     # eta_bp1_solutions, eta_bp1_cont = a.run(starting_point='BP1', origin=eta_hb2_cont, ISW=-1, name='eta_bp1', STOP={})
#     #
#     # # limit cycle continuation of hopf bifurcation in beta
#     # beta_hb1_solutions, beta_hb1_cont = a.run(starting_point='HB1', ICP=[8, 11], NMX=6000, origin=eta_cont, IPS=2,
#     #                                           RL0=-0.00001, RL1=1.0, DSMAX=1.0, name='beta_hb1', STOP='LP2')
#     # beta_hb2_solutions, beta_hb2_cont = a.run(starting_point='HB2', ICP=[8, 11], NMX=6000, origin=eta_cont, IPS=2,
#     #                                           RL0=-0.00001, RL1=1.0, DSMAX=1.0, name='beta_hb2', STOP='LP1')
#     # beta_bp1_solutions, beta_bp1_cont = a.run(starting_point='BP1', origin=beta_hb2_cont, ISW=-1, name='beta_bp1',
#     #                                           STOP={})
#     #
#     # if codim2:
#     #
#     #     # continue the limit cycle borders in eta_i and beta
#     #     j_hb1_solutions, j_hb1_cont = a.run(starting_point='HB1', c='qif2', ICP=[3, 4], NMX=20000, NDIM=n_dim,
#     #                                         DSMAX=10.0, NPAR=n_params, bidirectional=True, origin=eta_cont, RL0=0.0,
#     #                                         name='j_hb1')
#
#         #etai_beta_hb2_solutions, etai_beta_hb2_cont = a.run(starting_point='HB2', c='qif2', ICP=[8, 2], DSMAX=0.05,
#         #                                                    NMX=8000, bidirectional=True, origin=eta_cont, RL0=-0.001,
#         #                                                    name='etai_beta_hb2', NDIM=n_dim, RL1=1.0)
#
#         # continue the stable limit cycle borders in eta and alpha
#         #eta_alpha_hb1_solutions, eta_balpha_hb1_cont = a.run(starting_point='HB1', c='qif2', ICP=[7, 1], DSMAX=0.05,
#         #                                                     NMX=8000, bidirectional=True, origin=eta_cont, RL0=-0.001,
#         #                                                     name='eta_alpha_hb1', NDIM=n_dim, RL1=1.0)
#         #eta_alpha_hb2_solutions, eta_alpha_hb2_cont = a.run(starting_point='HB2', c='qif2', ICP=[7, 1], DSMAX=0.05,
#         #                                                    NMX=8000, bidirectional=True, origin=eta_cont, RL0=-0.001,
#         #                                                    name='eta_alpha_hb2', NDIM=n_dim, RL1=1.0)
#
#         # limit cycle continuation of hopf fold bifurcation in beta
#         #beta_hb_solutions, beta_hb_cont = a.run(starting_point='LP2', c='qif', ICP=8, NDIM=n_dim, NPAR=n_params,
#         #                                        DSMAX=0.01, NMX=10000, RL0=-0.001, RL1=1.0, name='beta_hb', ILP=0,
#         #                                        origin=eta_hb_cont, EPSL=1e-6, EPSU=1e-6, EPSS=1e-4, NTST=400, IPS=2)
#
#         # limit cycle continuation of hopf fold bifurcation in alpha
#         #alpha_hb_solutions, alpha_hb_cont = a.run(starting_point='LP2', c='qif', ICP=7, NDIM=n_dim, NTST=400, ILP=0,
#         #                                          NPAR=n_params, DSMAX=0.01, NMX=10000, RL0=-0.001, name='alpha_hb',
#         #                                          RL1=1.0, origin=eta_hb_cont, EPSL=1e-6, EPSU=1e-6, EPSS=1e-4, IPS=2)
#
#
