from pyauto import PyAuto
import numpy as np
import matplotlib.pyplot as plt

#####################################
# bifurcation analysis of qif model #
#####################################

# config
n_dim = 46
n_params = 17
a = PyAuto("auto_files")

# initial continuation in time
##############################

t_sols, t_cont = a.run(e='qif_stn_gpe', c='ivp', ICP=14, DS=5e-2, DSMIN=1e-4, DSMAX=1.0, NMX=1000000, name='t',
                       UZR={14: 10000.0}, STOP={'UZ1'}, NDIM=n_dim, NPAR=n_params)

# continuation in k_i
#####################

# step 1: codim 1 investigation
c3_sols, c3_cont = a.run(starting_point='UZ1', c='qif', ICP=10, NPAR=n_params, name='k_i', NDIM=n_dim,
                         RL0=0.0, RL1=10.0, origin=t_cont, bidirectional=True, NMX=6000, DSMAX=0.005)

# step 2: codim 1 continuation of limit cycle found in step 1
#c3_lc_sols, c3_lc_cont = a.run(starting_point='HB1', c='qif2b', ICP=[10, 11], DSMAX=0.1, NMX=6000,
#                               NPAR=n_params, name='k_i_lc', origin=c3_cont, NDIM=n_dim, RL0=0.0,
#                               RL1=10.0)

# step 3: codim 2 investigation of hopf bifurcation found in step 1
c3_d2_sols, c3_d2_cont = a.run(starting_point='HB1', c='qif2', ICP=[9, 10], DSMAX=0.1, NMX=10000,
                               NPAR=n_params, name='k/k_i', origin=c3_cont, NDIM=n_dim,
                               bidirectional=True, RL0=0.0, RL1=10.0)
c3_d2_2_sols, c3_d2_2_cont = a.run(starting_point='HB1', c='qif2', ICP=[17, 10], DSMAX=0.1, NMX=10000,
                                   NPAR=n_params, name='alpha/k_i', origin=c3_cont, NDIM=n_dim,
                                   bidirectional=True, RL0=0.0, RL1=1.0)

# step 4: codim 2 continuation of limit cycle found in step 1
c3_lc_d2_sols, c3_lc_d2_cont = a.run(starting_point='HB1', c='qif2b', ICP=[17, 10, 11], DSMAX=0.01, NMX=1000,
                                     NPAR=n_params, name='k_i_alpha_lc', origin=c3_cont, NDIM=n_dim, RL0=0.0,
                                     RL1=1.0, EPSL=1e-06, EPSU=1e-06, EPSS=1e-04, DSMIN=1e-8, STOP={'BP2'})

# continuation in alpha
#######################

# step 1: codim 1 investigation
c0_sols, c0_cont = a.run(starting_point='UZ1', c='qif', ICP=17, NPAR=n_params, name='alpha', NDIM=n_dim,
                         RL0=0.0, RL1=1.0, origin=t_cont, bidirectional=True, NMX=8000, DSMAX=0.01)

# step 2: codim 2 investigation of fold found in step 1
c0_d2_sols, c0_d2_cont = a.run(starting_point='LP1', c='qif2', ICP=[3, 17], DSMAX=0.1, NMX=10000,
                               NPAR=n_params, name='eta_str/alpha', origin=c0_cont, NDIM=n_dim,
                               bidirectional=True, RL0=-80.0, RL1=0.0)
c0_d2_2_sols, c0_d2_2_cont = a.run(starting_point='LP1', c='qif2', ICP=[9, 17], DSMAX=0.1, NMX=10000,
                                   NPAR=n_params, name='k/alpha', origin=c0_cont, NDIM=n_dim,
                                   bidirectional=True, RL0=0.0, RL1=10.0)
c0_d2_3_sols, c0_d2_3_cont = a.run(starting_point='LP1', c='qif2', ICP=[10, 17], DSMAX=0.1, NMX=10000,
                                   NPAR=n_params, name='k_i/alpha', origin=c0_cont, NDIM=n_dim,
                                   bidirectional=True, RL0=0.0, RL1=10.0)

# continuation in eta_str
#########################

# step 1: codim 1 investigation
c1_sols, c1_cont = a.run(starting_point='UZ1', c='qif', ICP=3, NPAR=n_params, name='eta_str', NDIM=n_dim,
                         RL0=-80.0, RL1=0.0, origin=t_cont, bidirectional=True, NMX=6000, DSMAX=0.05)

# step 2: codim 1 continuation of limit cycle found in step 1
#c1_lc_sols, c1_lc_cont = a.run(starting_point='HB1', c='qif2b', ICP=[3, 11], DSMAX=0.1, NMX=1000,
#                               NPAR=n_params, name='eta_str_lc', origin=c1_cont, NDIM=n_dim, RL0=-80.0,
#                               RL1=0.0)

# step 3: codim 2 investigations of hopf bifurcation found in step 1
c1_d2_sols, c1_d2_cont = a.run(starting_point='HB1', c='qif2', ICP=[7, 3], DSMAX=0.1, NMX=10000,
                               NPAR=n_params, name='k_ie/eta_str', origin=c1_cont, NDIM=n_dim,
                               bidirectional=True, RL0=0.0, RL1=200.0)
c1_d2_2_sols, c1_d2_2_cont = a.run(starting_point='HB1', c='qif2', ICP=[17, 3], DSMAX=0.1, NMX=10000,
                                   NPAR=n_params, name='alpha/eta_str', origin=c1_cont, NDIM=n_dim,
                                   bidirectional=True, RL0=0.0, RL1=1.0)

# step 4: codim 2 continuation of limit cycle found in step 1
c1_lc_d2_sols, c1_lc_d2_cont = a.run(starting_point='HB1', c='qif2b', ICP=[3, 17, 11], DSMAX=0.1, NMX=6000,
                                     NPAR=n_params, name='eta_str_alpha_lc', origin=c1_cont, NDIM=n_dim, RL0=-80.0,
                                     RL1=0.0, STOP={'TR1'})

# continuation in k
###################

# step 1: codim 1 investigation
c2_sols, c2_cont = a.run(starting_point='UZ1', c='qif', ICP=9, NPAR=n_params, name='k', NDIM=n_dim,
                         RL0=0.0, RL1=10.0, origin=t_cont, bidirectional=True, NMX=6000, DSMAX=0.005)

# step 2: codim 1 continuation of limit cycle found in step 1
#c2_lc_sols, c2_lc_cont = a.run(starting_point='HB1', c='qif2b', ICP=[9, 11], DSMAX=0.1, NMX=6000,
#                               NPAR=n_params, name='k_lc', origin=c2_cont, NDIM=n_dim, RL0=0.0,
#                               RL1=10.0)

# step 3: codim 2 investigations of hopf bifurcation found in step 1
c2_d2_sols, c2_d2_cont = a.run(starting_point='HB1', c='qif2', ICP=[10, 9], DSMAX=0.1, NMX=10000,
                               NPAR=n_params, name='k_i/k', origin=c2_cont, NDIM=n_dim,
                               bidirectional=True, RL0=0.0, RL1=10.0)
c2_d2_2_sols, c2_d2_2_cont = a.run(starting_point='HB1', c='qif2', ICP=[17, 9], DSMAX=0.1, NMX=10000,
                                   NPAR=n_params, name='alpha/k', origin=c2_cont, NDIM=n_dim,
                                   bidirectional=True, RL0=0.0, RL1=1.0)

# step 4: codim 2 continuation of limit cycle found in step 1
c2_lc_d2_sols, c2_lc_d2_cont = a.run(starting_point='HB1', c='qif2b', ICP=[9, 17, 11], DSMAX=0.1, NMX=6000,
                                     NPAR=n_params, name='k_alpha_lc', origin=c2_cont, NDIM=n_dim, RL0=0.0,
                                     RL1=10.0)

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
