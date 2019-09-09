import os
import numpy as np
from pyauto import PyAuto

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
codim2 = True
period_mapping = False
n_grid_points = 100

###################################
# parameter continuations in auto #
###################################

a = PyAuto("auto_files")

# initial continuation in the adaptation strength alpha
alpha_0 = [0.005, 0.01, 0.02, 0.04, 0.08]
alpha_solutions, alpha_cont = a.run(e='qif_biexp_mult', c='qif', ICP=3, UZR={3: alpha_0},
                                    STOP=['UZ' + str(len(alpha_0))], DSMAX=0.005, NMX=4000, name='s0')

# principle continuation in eta
###############################

# continue in eta for each adaptation rate alpha
solutions_eta = []
solutions_eta.append(a.run(starting_point='EP1', ICP=1, DSMAX=0.005, RL1=0.0, RL0=-12.0, NMX=2000, origin=alpha_cont,
                           bidirectional=True, name='eta_0'))

for i, (point, point_info) in enumerate(alpha_solutions.items()):
    if 'UZ' in point_info['bifurcation']:
        solutions_eta.append(a.run(starting_point=point, ICP=1, DSMAX=0.005, RL1=0.0, RL0=-12.0, NMX=4000,
                                   origin=alpha_cont, bidirectional=True, name=f"eta_{i+1}"))

# choose a continuation in eta to run further continuations on
eta_points, eta_cont = solutions_eta[3]

if codim1:

    # limit cycle continuation of hopf bifurcations in eta
    eta_hb1_solutions, eta_hb1_cont = a.run(starting_point='HB1', c='qif2b', ICP=1, DSMAX=0.1, NMX=2000,
                                            origin=eta_cont, name='eta_hb1')
    eta_hb2_solutions, eta_hb2_cont = a.run(starting_point='HB2', c='qif2b', ICP=1, DSMAX=0.1, NMX=2000,
                                            origin=eta_cont, name='eta_hb2')
    tau2_hb2_solutions, tau2_hb2_cont = a.run(starting_point='HB2', c='qif2b', ICP=5, DSMAX=0.1, NMX=300, RL0=0.0,
                                              origin=eta_cont, name='tau2_hb2')

    if codim2:

        # continuation in eta and tau2
        ##############################

        # continue the limit cycle borders in tau2 and eta
        eta_tau2_hb2_solutions, eta_tau2_hb2_cont = a.run(starting_point='HB2', c='qif2', ICP=[1, 5], DSMAX=0.01,
                                                          NMX=6000, bidirectional=True, origin=eta_cont,
                                                          name='eta_tau2_hb2')

        # continue the stable region of the limit cycle
        tau2s = np.round(np.linspace(2e-2, 25.0, n_grid_points)[::-1], decimals=4).tolist()
        eta_tau2_hb2_lc_solutions, eta_tau2_hb2_lc_cont = a.run(starting_point='LP2', c='qif3', ICP=[1, 11, 5],
                                                                NMX=4000, DSMAX=0.01, UZR={5: tau2s}, STOP=[],
                                                                origin=eta_hb2_cont, bidirectional=True,
                                                                name='eta_tau2_hb2b')

        # continue the fold bifurcation in eta and tau2
        eta_tau2_lp2_solution, eta_tau2_lp2_cont = a.run(starting_point='LP2', c='qif2', ICP=[1, 5], DSMAX=0.01,
                                                         NMX=4000, origin=eta_cont, bidirectional=True,
                                                         name='eta_tau2_lp2')

        # get limit cycle periods
        if period_mapping:

            etas = np.round(np.linspace(-6, -3, n_grid_points), decimals=4).tolist()
            period_solutions = np.zeros((len(tau2s), len(etas)))
            for point, point_info in eta_tau2_hb2_lc_solutions.items():
                if np.round(point_info['PAR(5)'], decimals=4) in tau2s:
                    solution_tmp, cont_tmp = a.run(starting_point=point, c='qif', ICP=1, UZR={1: etas}, STOP={},
                                                   EPSL=1e-6, EPSU=1e-6, EPSS=1e-4, ILP=0, ISP=0, IPS=2, DSMAX=0.05,
                                                   DS='-', origin=eta_tau2_hb2_lc_cont, get_period=True)
                    for point_tmp, point_info_tmp in solution_tmp.items():
                        if 'UZ' in point_info_tmp['bifurcation']:
                            idx_c = np.argwhere(np.round(point_info_tmp['PAR(1)'], decimals=4) == etas)
                            idx_r = np.argwhere(np.round(point_info_tmp['PAR(5)'], decimals=4) == tau2s)
                            period_solutions[idx_r, idx_c] = point_info_tmp['period']

        # continuation in eta and tau1
        ###############################

        # continue the limit cycle borders in eta and tau1
        eta_tau1_hb2_solution, eta_tau1_hb2_cont = a.run(starting_point='HB2', c='qif2', ICP=[1, 4], DSMAX=0.01,
                                                         NMX=4000, origin=eta_cont, bidirectional=True)

        # continue the stable region of the limit cycle
        eta_tau1_lp2_solution, eta_tau1_lp2_cont = a.run(starting_point='LP2', c='qif3', ICP=[1, 11, 4], NMX=4000,
                                                         DSMAX=0.01, STOP='LP2', origin=eta_hb2_cont,
                                                         bidirectional=True)

        # continuation in tau1 and tau2
        ###############################

        # continue the limit cycle borders in alpha and tau1
        a.run(starting_point='HB2', c='qif2', ICP=[5, 4], DSMAX=0.01, NMX=4000, RL0=0.0, origin=eta_cont)
        tau2_tau1_hb2_solution, tau2_tau1_hb2_cont = a.run(starting_point='HB2', c='qif2', ICP=[5, 4], DSMAX=0.01,
                                                           NMX=2000, RL0=0.0, DS='-', origin=eta_cont)

        # continue the stable region of the limit cycle
        a.run(starting_point='LP2', c='qif3', ICP=[5, 11, 4], NMX=1800, RL0=0.0, DSMAX=0.01, origin=tau2_hb2_cont)
        tau2_tau1_hb2_lc_solution, tau2_tau1_hb2_lc_cont = a.run(starting_point='LP2', c='qif3', ICP=[5, 11, 4], DS='-',
                                                                 NMX=8000, RL0=0.0, DSMAX=0.01, origin=tau2_hb2_cont)

################
# save results #
################

fname = 'biexp_mult.hdf5'
a.to_h5py(fname)

if period_mapping:
    period_solutions.tofile(f"biexp_mult_period")
