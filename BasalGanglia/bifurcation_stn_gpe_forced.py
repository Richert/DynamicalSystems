from pyauto import PyAuto
import numpy as np
import matplotlib.pyplot as plt

#####################################
# bifurcation analysis of qif model #
#####################################

# config
n_dim = 48
n_params = 32
a = PyAuto("auto_files")
fname = '../results/qif_stn_gpe_forced.pkl'

# initial continuation in beta (driver amplitude)
#################################################

# step 1: codim 1 investigation
init_sols, init_cont = a.run(e='stn_gpe_forced', c='qif_lc', ICP=[31, 11], NPAR=n_params, name='beta', NDIM=n_dim,
                             RL0=0.0, RL1=70.0, bidirectional=True, NMX=6000, DSMAX=0.5, STOP={'UZ5'},
                             UZR={31: [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]})

starting_point = 'UZ4'
starting_cont = init_cont

# continuation in alpha
#######################

# step 1: codim 1 investigation
c0_sols, c0_cont = a.run(starting_point=starting_point, c='qif_lc', ICP=[26, 11], NPAR=n_params, name='alpha',
                         NDIM=n_dim, RL0=0.0, RL1=1.0, origin=starting_cont, bidirectional=True, NMX=600, DSMAX=0.1)

# step 2: codim 2 investigation of fold found in step 1
c0_d2_sols, c0_d2_cont = a.run(starting_point='LP1', c='qif3', ICP=[3, 26, 11], DSMAX=0.1, NMX=1000,
                               NPAR=n_params, name='eta_str/alpha', origin=c0_cont, NDIM=n_dim,
                               bidirectional=True, RL0=-80.0, RL1=0.0)
c0_d2_2_sols, c0_d2_2_cont = a.run(starting_point='LP1', c='qif3', ICP=[9, 26, 11], DSMAX=0.1, NMX=1000,
                                   NPAR=n_params, name='k/alpha', origin=c0_cont, NDIM=n_dim,
                                   bidirectional=True, RL0=0.0, RL1=10.0)
# c0_d2_3_sols, c0_d2_3_cont = a.run(starting_point='LP1', c='qif2', ICP=[10, 26], DSMAX=0.1, NMX=6000,
#                                    NPAR=n_params, name='k_i/alpha', origin=c0_cont, NDIM=n_dim,
#                                    bidirectional=True, RL0=0.0, RL1=10.0)

# step 3: codim 2 investigation of hopf codim 2 bifurcations found in step 2

# save results
kwargs = dict()
a.to_file(fname, **kwargs)

# continuation in k
###################

# step 1: codim 1 investigation
k_vals = [0.6, 0.8, 1.2, 1.4, 1.6, 1.8, 2.0]
c2_sols, c2_cont = a.run(starting_point=starting_point, c='qif_lc', ICP=[9, 11], NPAR=n_params, name='k', NDIM=n_dim,
                         RL0=0.5, RL1=2.1, origin=starting_cont, bidirectional=True, NMX=600, DSMAX=0.05,
                         UZR={9: k_vals}, STOP={})

# step 2: continue each user point from step 1 in eta_str
# for i, k in enumerate(k_vals):
#     s_tmp, c_tmp = a.run(starting_point=f'UZ{i+1}', c='qif_lc', ICP=[3, 11], NPAR=n_params, name=f'eta_str_{i+1}',
#                          NDIM=n_dim, RL0=-100.0, RL1=0.0, origin=c2_cont, bidirectional=True, NMX=600, DSMAX=0.5)
#     bifs = [s['bifurcation'] for s in s_tmp.values()]
    # if 'HB' in bifs:
    #     s_lc_tmp, c_lc_tmp = a.run(starting_point='HB1', c='qif2b', ICP=[3, 11], DSMAX=0.1, NMX=1000,
    #                                NPAR=n_params, name=f'eta_str_{i+1}_lc', origin=c_tmp, NDIM=n_dim, RL0=-100.0,
    #                                RL1=0.0, STOP={'BP1'}, NPR=10)
    #     if bifs.count('HB') > 2:
    #         s_lc_tmp, c_lc_tmp = a.run(starting_point='HB3', c='qif2b', ICP=[3, 11], DSMAX=0.1, NMX=1000,
    #                                    NPAR=n_params, name=f'eta_str_{i + 1}_hb2_lc', origin=c_tmp, NDIM=n_dim,
    #                                    RL0=-100.0, RL1=0.0, STOP={'BP1'}, NPR=10)

# save results
kwargs = dict()
a.to_file(fname, **kwargs)

# continuation in eta_str
#########################

# step 1: codim 1 investigation
c1_sols, c1_cont = a.run(starting_point=starting_point, c='qif_lc', ICP=[3, 11], NPAR=n_params, name='eta_str',
                         NDIM=n_dim, RL0=-100.0, RL1=0.0, origin=starting_cont, bidirectional=True, NMX=800, DSMAX=0.05)

# # step 2: codim 1 continuation of limit cycles found in step 1
# c1_lc_sols, c1_lc_cont = a.run(starting_point='HB1', c='qif2b', ICP=[3, 11], DSMAX=0.05, NMX=2000,
#                                NPAR=n_params, name='eta_str_lc', origin=c1_cont, NDIM=n_dim, RL0=-100.0,
#                                RL1=0.0, STOP={'BP1'}, NPR=10)
# c1_lc2_sols, c1_lc2_cont = a.run(starting_point='HB3', c='qif2b', ICP=[3, 11], DSMAX=0.05, NMX=2000,
#                                  NPAR=n_params, name='eta_str_lc2', origin=c1_cont, NDIM=n_dim, RL0=-100.0,
#                                  RL1=0.0, STOP={'BP1'}, NPR=10)
#
# # step 3: codim 2 continuation of limit cycles found in step 1
# c1_lc_d2_sols, c1_lc_d2_cont = a.run(starting_point='HB2', c='qif2b', ICP=[17, 3, 11], DSMAX=0.05, NMX=2000,
#                                      NPAR=n_params, name='eta_str_delta_lc', origin=c1_cont, NDIM=n_dim, RL0=0.0,
#                                      RL1=0.1, STOP={'TR1', 'BP3'}, NPR=10)
# c1_lc2_d2_sols, c1_lc2_d2_cont = a.run(starting_point='HB2', c='qif2b', ICP=[9, 3, 11], DSMAX=0.05, NMX=2000,
#                                        NPAR=n_params, name='eta_str_k_lc', origin=c1_cont, NDIM=n_dim, RL0=-100.0,
#                                        RL1=0.0, STOP={'TR1', 'BP3'}, NPR=10)
#
# # step 4: codim 2 investigations of hopf bifurcation found in step 1
# c1_d2_sols, c1_d2_cont = a.run(starting_point='HB1', c='qif2', ICP=[9, 3], DSMAX=0.1, NMX=6000,
#                                NPAR=n_params, name='k/eta_str', origin=c1_cont, NDIM=n_dim,
#                                bidirectional=True, RL0=0.0, RL1=10.0)
# c1_d2_2_sols, c1_d2_2_cont = a.run(starting_point='HB1', c='qif2', ICP=[17, 3], DSMAX=0.1, NMX=6000,
#                                    NPAR=n_params, name='alpha/eta_str', origin=c1_cont, NDIM=n_dim,
#                                    bidirectional=True, RL0=0.0, RL1=1.0)
# c1_d2_3_sols, c1_d2_3_cont = a.run(starting_point='HB1', c='qif2', ICP=[10, 3], DSMAX=0.1, NMX=6000,
#                                    NPAR=n_params, name='k_i/eta_str', origin=c1_cont, NDIM=n_dim,
#                                    bidirectional=True, RL0=0.0, RL1=10.0)
# c1_d2_5_sols, c1_d2_5_cont = a.run(starting_point='HB3', c='qif2', ICP=[9, 3], DSMAX=0.1, NMX=6000,
#                                    NPAR=n_params, name='k/eta_str_2', origin=c1_cont, NDIM=n_dim,
#                                    bidirectional=True, RL0=0.0, RL1=10.0)
# c1_d2_6_sols, c1_d2_6_cont = a.run(starting_point='HB3', c='qif2', ICP=[17, 3], DSMAX=0.1, NMX=6000,
#                                    NPAR=n_params, name='alpha/eta_str_2', origin=c1_cont, NDIM=n_dim,
#                                    bidirectional=True, RL0=0.0, RL1=1.0)
# c1_d2_7_sols, c1_d2_7_cont = a.run(starting_point='HB3', c='qif2', ICP=[10, 3], DSMAX=0.1, NMX=6000,
#                                    NPAR=n_params, name='k_i/eta_str_2', origin=c1_cont, NDIM=n_dim,
#                                    bidirectional=True, RL0=0.0, RL1=10.0)
#
# # step 5: codim 2 continuation of zero-hopf bifurcation found in step 4
# c1_zh_d2_sols, c1_zh_d2_cont = a.run(starting_point='ZH1', c='qif2', ICP=[9, 3], DSMAX=0.1, NMX=6000,
#                                      NPAR=n_params, name='k/eta_str_zh', origin=c1_d2_cont, NDIM=n_dim,
#                                      bidirectional=True, RL0=0.0, RL1=10.0)
# c1_zh_d2_2_sols, c1_zh_d2_2_cont = a.run(starting_point='ZH1', c='qif2', ICP=[17, 3], DSMAX=0.1, NMX=6000,
#                                          NPAR=n_params, name='alpha/eta_str_zh', origin=c1_d2_cont, NDIM=n_dim,
#                                          bidirectional=True, RL0=0.0, RL1=1.0)
# c1_zh_dh2_3_sols, c1_zh_dh2_3_cont = a.run(starting_point='ZH1', c='qif2', ICP=[17, 9], DSMAX=0.1, NMX=6000,
#                                            NPAR=n_params, name='alpha/k_zh', origin=c1_d2_cont, NDIM=n_dim,
#                                            bidirectional=True, RL0=0.0, RL1=1.0)

# save results
kwargs = dict()
a.to_file(fname, **kwargs)

# # step 2: codim 1 continuation of limit cycle found in step 1
# c2_lc_sols, c2_lc_cont = a.run(starting_point='HB1', c='qif2b', ICP=[9, 11], DSMAX=0.1, NMX=2000,
#                                NPAR=n_params, name='k_lc', origin=c2_cont, NDIM=n_dim, RL0=0.0,
#                                RL1=10.0, STOP={'BP2'}, NPR=10)
#
# # step 3: codim 2 investigations of hopf bifurcation found in step 1
# c2_d2_sols, c2_d2_cont = a.run(starting_point='HB1', c='qif2', ICP=[10, 9], DSMAX=0.1, NMX=6000,
#                                NPAR=n_params, name='k_i/k', origin=c2_cont, NDIM=n_dim,
#                                bidirectional=True, RL0=0.0, RL1=10.0)
# c2_d2_2_sols, c2_d2_2_cont = a.run(starting_point='HB1', c='qif2', ICP=[17, 9], DSMAX=0.1, NMX=6000,
#                                    NPAR=n_params, name='alpha/k', origin=c2_cont, NDIM=n_dim,
#                                    bidirectional=True, RL0=0.0, RL1=0.1)
# c2_d2_3_sols, c2_d2_3_cont = a.run(starting_point='HB1', c='qif2', ICP=[18, 9], DSMAX=0.1, NMX=6000,
#                                    NPAR=n_params, name='delta/k', origin=c2_cont, NDIM=n_dim,
#                                    bidirectional=True, RL0=0.0, RL1=2.0)
# c2_d2_4_sols, c2_d2_4_cont = a.run(starting_point='HB1', c='qif2', ICP=[3, 9], DSMAX=0.1, NMX=6000,
#                                    NPAR=n_params, name='eta_str/k', origin=c2_cont, NDIM=n_dim,
#                                    bidirectional=True, RL0=-100.0, RL1=0.0)
#
# # step 4: codim 2 continuation of limit cycle found in step 1
# c2_lc_d2_sols, c2_lc_d2_cont = a.run(starting_point='HB1', c='qif2b', ICP=[9, 3, 11], DSMAX=0.1, NMX=2000,
#                                      NPAR=n_params, name='k/eta_str_lc', origin=c2_cont, NDIM=n_dim, RL0=0.0,
#                                      RL1=10.0, STOP={'BP2'}, NPR=10)
# c2_lc2_d2_sols, c2_lc2_d2_cont = a.run(starting_point='HB1', c='qif2b', ICP=[9, 17, 11], DSMAX=0.05, NMX=2000,
#                                        NPAR=n_params, name='k/alpha_lc', origin=c2_cont, NDIM=n_dim, RL0=0.0,
#                                        RL1=10.0, STOP={'BP2'}, NPR=10)
# c2_lc3_d2_sols, c2_lc3_d2_cont = a.run(starting_point='HB1', c='qif2b', ICP=[9, 18, 11], DSMAX=0.05, NMX=2000,
#                                        NPAR=n_params, name='k/delta_lc', origin=c2_cont, NDIM=n_dim, RL0=0.0,
#                                        RL1=10.0, STOP={'BP2'}, NPR=10)

# # continuation in k_i
# #####################
#
# # # step 1: codim 1 investigation
# # c3_sols, c3_cont = a.run(starting_point='UZ1', c='qif', ICP=10, NPAR=n_params, name='k_i', NDIM=n_dim,
# #                          RL0=0.0, RL1=10.0, origin=t_cont, bidirectional=True, NMX=6000, DSMAX=0.005)
# #
# # # step 2: codim 1 continuation of limit cycle found in step 1
# # c3_lc_sols, c3_lc_cont = a.run(starting_point='HB1', c='qif2b', ICP=[10, 11], DSMAX=0.1, NMX=6000,
# #                                NPAR=n_params, name='k_i_lc', origin=c3_cont, NDIM=n_dim, RL0=0.0,
# #                                RL1=10.0)
# #
# # # step 3: codim 2 investigation of hopf bifurcation found in step 1
# # c3_d2_sols, c3_d2_cont = a.run(starting_point='HB1', c='qif2', ICP=[9, 10], DSMAX=0.1, NMX=10000,
# #                                NPAR=n_params, name='k/k_i', origin=c3_cont, NDIM=n_dim,
# #                                bidirectional=True, RL0=0.0, RL1=10.0)
# # c3_d2_2_sols, c3_d2_2_cont = a.run(starting_point='HB1', c='qif2', ICP=[17, 10], DSMAX=0.1, NMX=10000,
# #                                    NPAR=n_params, name='alpha/k_i', origin=c3_cont, NDIM=n_dim,
# #                                    bidirectional=True, RL0=0.0, RL1=1.0)
# #
# # # step 4: codim 2 continuation of limit cycle found in step 1
# # c3_lc_d2_sols, c3_lc_d2_cont = a.run(starting_point='HB1', c='qif2b', ICP=[17, 10, 11], DSMAX=0.01, NMX=1000,
# #                                      NPAR=n_params, name='k_i_alpha_lc', origin=c3_cont, NDIM=n_dim, RL0=0.0,
# #                                      RL1=1.0, EPSL=1e-06, EPSU=1e-06, EPSS=1e-04, DSMIN=1e-8, STOP={'BP2'})

# save results
kwargs = dict()
a.to_file(fname, **kwargs)

# continuation in delta
#######################

# # step 1: codim 1 investigation
# c4_sols, c4_cont = a.run(starting_point=starting_point, c='qif', ICP=18, NPAR=n_params, name='delta', NDIM=n_dim,
#                          RL0=0.0, RL1=2.0, origin=starting_cont, bidirectional=True, NMX=6000, DSMAX=0.005)
#
# # step 2: codim 1 continuation of limit cycle found in step 1
# c4_lc_sols, c4_lc_cont = a.run(starting_point='HB1', c='qif2b', ICP=[18, 11], DSMAX=0.05, NMX=2000,
#                                NPAR=n_params, name='delta_lc', origin=c4_cont, NDIM=n_dim, RL0=0.0,
#                                RL1=2.0, STOP={'BP2'}, NPR=10)
#
# # step 3: codim 2 investigation of hopf bifurcation found in step 1
# c4_d2_sols, c4_d2_cont = a.run(starting_point='HB1', c='qif2', ICP=[9, 18], DSMAX=0.1, NMX=6000,
#                                NPAR=n_params, name='k/delta', origin=c4_cont, NDIM=n_dim,
#                                bidirectional=True, RL0=0.0, RL1=10.0)
# c4_d2_2_sols, c4_d2_2_cont = a.run(starting_point='HB1', c='qif2', ICP=[17, 18], DSMAX=0.05, NMX=6000,
#                                    NPAR=n_params, name='alpha/delta', origin=c4_cont, NDIM=n_dim,
#                                    bidirectional=True, RL0=0.0, RL1=0.1)
# c4_d2_3_sols, c4_d2_3_cont = a.run(starting_point='HB1', c='qif2', ICP=[3, 18], DSMAX=0.1, NMX=6000,
#                                    NPAR=n_params, name='eta_str/delta', origin=c4_cont, NDIM=n_dim,
#                                    bidirectional=True, RL0=-100.0, RL1=0.0)
# c4_d2_4_sols, c4_d2_4_cont = a.run(starting_point='HB1', c='qif2', ICP=[10, 18], DSMAX=0.1, NMX=6000,
#                                    NPAR=n_params, name='k_i/delta', origin=c4_cont, NDIM=n_dim,
#                                    bidirectional=True, RL0=0.0, RL1=10.0)
#
# # step 4: codim 2 continuation of limit cycle found in step 1
# c4_lc_d2_sols, c4_lc_d2_cont = a.run(starting_point='HB1', c='qif2b', ICP=[18, 3, 11], DSMAX=0.05, NMX=2000,
#                                      NPAR=n_params, name='delta/eta_str_lc', origin=c4_cont, NDIM=n_dim, RL0=0.0,
#                                      RL1=2.0, STOP={'BP2'}, NPR=10)
# c4_lc2_d2_sols, c4_lc2_d2_cont = a.run(starting_point='HB1', c='qif2b', ICP=[18, 9, 11], DSMAX=0.05, NMX=2000,
#                                        NPAR=n_params, name='delta/eta_str_lc', origin=c4_cont, NDIM=n_dim, RL0=0.0,
#                                        RL1=10.0, STOP={'BP2'}, NPR=10)
# c4_lc3_d2_sols, c4_lc3_d2_cont = a.run(starting_point='HB1', c='qif2b', ICP=[18, 17, 11], DSMAX=0.05, NMX=2000,
#                                        NPAR=n_params, name='delta/alpha_lc', origin=c4_cont, NDIM=n_dim, RL0=0.0,
#                                        RL1=2.0, STOP={'BP2'}, NPR=10, NTST=400)

# save results
kwargs = dict()
a.to_file(fname, **kwargs)