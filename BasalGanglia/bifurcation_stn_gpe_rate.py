from pyauto import PyAuto
import numpy as np
import matplotlib.pyplot as plt

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

with parameters:

 PAR1: eta = -10.0
 PAR2: J = 15.0
 PAR3: alpha = 0.0
 PAR4: tau = 1.0
 PAR5: D = 2.0

"""

# configuration
codim1 = True
codim2 = True
n_grid_points = 100
n_dim = 10
n_params = 20
cont_idx = 0

###################################
# parameter continuations in auto #
###################################

a = PyAuto("auto_files")

# principle continuation in eta
###############################

# continue in eta_e
etas = [0.2, 0.4, 0.6, 0.8]
n = len(etas)
etae_solutions, etae_cont = a.run(e='stn_gpe_rate', c='qif', ICP=1, DSMAX=0.005, NMX=6000, NPAR=n_params,
                                  name='eta_e', NDIM=n_dim, STOP={}, RL0=0.0, RL1=1.0, DS='-', UZR={1: etas})

# continue in eta_i
solutions = []
i = 1
for point, point_info in etae_solutions.items():
    if 'UZ' in point_info['bifurcation']:
        solutions.append(a.run(starting_point=f'UZ{i}', ICP=2, DSMAX=0.005, NMX=6000, NPAR=n_params,
                               name=f'eta_{i}', NDIM=n_dim, STOP={}, RL0=-20.0, RL1=0.0, origin=etae_cont,
                               bidirectional=True))
        i += 1

# choose a continuation in eta to run further continuations on
eta_points, eta_cont = solutions[cont_idx]

if codim1:

    # limit cycle continuation of hopf bifurcations in eta_i
    eta_hb1_solutions, eta_hb1_cont = a.run(starting_point='HB1', ICP=[2, 11], NMX=8000, origin=eta_cont,
                                            name='eta_i_hb', IPS=2, DSMAX=0.1)

    # limit cycle continuation of hopf bifurcations in k_e
    ke_hb1_solutions, ke_hb1_cont = a.run(starting_point='HB1', ICP=[3, 11], NMX=8000, origin=eta_cont,
                                          name='k_e_hb', IPS=2, DSMAX=0.1)

    # limit cycle continuation of hopf bifurcations in k_i
    ki_hb1_solutions, ki_hb1_cont = a.run(starting_point='HB1', ICP=[4, 11], NMX=8000, origin=eta_cont,
                                          name='k_i_hb', IPS=2, DSMAX=0.1)

    # limit cycle continuation of hopf bifurcations in alpha
    alpha_solutions, alpha_cont = a.run(starting_point='HB1', ICP=[5, 11], NMX=8000, origin=eta_cont,
                                        name='alpha_hb', IPS=2, DSMAX=0.1)

    # eta_bp1_solutions, eta_bp1_cont = a.run(starting_point='BP1', origin=eta_hb2_cont, ISW=-1, name='eta_bp1', STOP={})
    #
    # # limit cycle continuation of hopf bifurcation in beta
    # beta_hb1_solutions, beta_hb1_cont = a.run(starting_point='HB1', ICP=[8, 11], NMX=6000, origin=eta_cont, IPS=2,
    #                                           RL0=-0.00001, RL1=1.0, DSMAX=1.0, name='beta_hb1', STOP='LP2')
    # beta_hb2_solutions, beta_hb2_cont = a.run(starting_point='HB2', ICP=[8, 11], NMX=6000, origin=eta_cont, IPS=2,
    #                                           RL0=-0.00001, RL1=1.0, DSMAX=1.0, name='beta_hb2', STOP='LP1')
    # beta_bp1_solutions, beta_bp1_cont = a.run(starting_point='BP1', origin=beta_hb2_cont, ISW=-1, name='beta_bp1',
    #                                           STOP={})
    #
    # if codim2:
    #
    #     # continue the limit cycle borders in eta_i and beta
    #     j_hb1_solutions, j_hb1_cont = a.run(starting_point='HB1', c='qif2', ICP=[3, 4], NMX=20000, NDIM=n_dim,
    #                                         DSMAX=10.0, NPAR=n_params, bidirectional=True, origin=eta_cont, RL0=0.0,
    #                                         name='j_hb1')

        #etai_beta_hb2_solutions, etai_beta_hb2_cont = a.run(starting_point='HB2', c='qif2', ICP=[8, 2], DSMAX=0.05,
        #                                                    NMX=8000, bidirectional=True, origin=eta_cont, RL0=-0.001,
        #                                                    name='etai_beta_hb2', NDIM=n_dim, RL1=1.0)

        # continue the stable limit cycle borders in eta and alpha
        #eta_alpha_hb1_solutions, eta_balpha_hb1_cont = a.run(starting_point='HB1', c='qif2', ICP=[7, 1], DSMAX=0.05,
        #                                                     NMX=8000, bidirectional=True, origin=eta_cont, RL0=-0.001,
        #                                                     name='eta_alpha_hb1', NDIM=n_dim, RL1=1.0)
        #eta_alpha_hb2_solutions, eta_alpha_hb2_cont = a.run(starting_point='HB2', c='qif2', ICP=[7, 1], DSMAX=0.05,
        #                                                    NMX=8000, bidirectional=True, origin=eta_cont, RL0=-0.001,
        #                                                    name='eta_alpha_hb2', NDIM=n_dim, RL1=1.0)

        # limit cycle continuation of hopf fold bifurcation in beta
        #beta_hb_solutions, beta_hb_cont = a.run(starting_point='LP2', c='qif', ICP=8, NDIM=n_dim, NPAR=n_params,
        #                                        DSMAX=0.01, NMX=10000, RL0=-0.001, RL1=1.0, name='beta_hb', ILP=0,
        #                                        origin=eta_hb_cont, EPSL=1e-6, EPSU=1e-6, EPSS=1e-4, NTST=400, IPS=2)

        # limit cycle continuation of hopf fold bifurcation in alpha
        #alpha_hb_solutions, alpha_hb_cont = a.run(starting_point='LP2', c='qif', ICP=7, NDIM=n_dim, NTST=400, ILP=0,
        #                                          NPAR=n_params, DSMAX=0.01, NMX=10000, RL0=-0.001, name='alpha_hb',
        #                                          RL1=1.0, origin=eta_hb_cont, EPSL=1e-6, EPSU=1e-6, EPSS=1e-4, IPS=2)


################
# save results #
################

fname = '../results/eic.pkl'
kwargs = dict()
a.to_file(fname, **kwargs)
