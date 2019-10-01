from pyauto import PyAuto
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
period_mapping = False
n_grid_points = 100
n_dim = 4
n_params = 6
eta_cont_idx = 4

###################################
# parameter continuations in auto #
###################################

a = PyAuto("auto_files")

# initial continuation in the adaptation strength alpha
alpha_0 = [0.1, 0.2, 0.4, 0.8, 1.6]
alpha_solutions, alpha_cont = a.run(e='qif_alpha_add', c='qif', ICP=3, UZR={3: alpha_0}, NDIM=n_dim,
                                    STOP=['UZ' + str(len(alpha_0))], DSMAX=0.005, NMX=4000, name='s0')

# principle continuation in eta
###############################

# continue in eta for each adaptation rate alpha
solutions_eta = list()
solutions_eta.append(a.run(starting_point='EP1', ICP=1, DSMAX=0.005, RL0=-12.0, NMX=2000, origin=alpha_cont,
                           bidirectional=True, name='eta_0', NDIM=n_dim))

i = 1
for point, point_info in alpha_solutions.items():
    if 'UZ' in point_info['bifurcation']:
        solutions_eta.append(a.run(starting_point=point, ICP=1, DSMAX=0.005, RL0=-12.0, NMX=4000,
                                   origin=alpha_cont, bidirectional=True, name=f"eta_{i}", NDIM=n_dim))
        i += 1

# choose a continuation in eta to run further continuations on
eta_points, eta_cont = solutions_eta[eta_cont_idx]

if codim1:

    # limit cycle continuation of hopf bifurcations in eta
    eta_hb2_solutions, eta_hb2_cont = a.run(starting_point='HB2', c='qif2b', ICP=[1, 11], DSMAX=0.05, NMX=4000,
                                            origin=eta_cont, name='eta_hb2', NDIM=n_dim)

    # continuation in eta and alpha
    ###############################

    if codim2:

        # continue the limit cycle borders in alpha and eta
        eta_alpha_hb1_solutions, eta_alpha_hb1_cont = a.run(starting_point='HB1', c='qif2', ICP=[3, 1], DSMAX=0.005,
                                                            NMX=8000, bidirectional=True, origin=eta_cont, RL0=0.0,
                                                            name='eta_alpha_hb1', NDIM=n_dim)
        eta_alpha_hb2_solutions, eta_alpha_hb2_cont = a.run(starting_point='HB2', c='qif2', ICP=[3, 1], DSMAX=0.005,
                                                            NMX=8000, bidirectional=True, origin=eta_cont, RL0=0.0,
                                                            name='eta_alpha_hb2', NDIM=n_dim)

        # continue the fold borders in alpha and eta
        eta_alpha_lp1_solutions, eta_alpha_lp1_cont = a.run(starting_point='LP1', c='qif2', ICP=[3, 1], DSMAX=0.05,
                                                            NMX=8000, origin=eta_cont, name='eta_alpha_lp1', NDIM=n_dim,
                                                            bidirectional=True, RL0=0.0)

        # continue the first and second fold bifurcation of the limit cycle
        alphas = np.round(np.linspace(0., 2.0, 100)[::-1], decimals=4).tolist()
        eta_alpha_lp2_solutions, eta_alpha_lp2_cont = a.run(starting_point='LP1', c='qif3', ICP=[1, 11, 3],
                                                            NMX=12000, DSMAX=0.05, origin=eta_hb2_cont,
                                                            bidirectional=True, name='eta_alpha_lp2', NDIM=n_dim,
                                                            UZR={3: alphas}, STOP={})
        eta_alpha_lp3_solutions, eta_alpha_lp3_cont = a.run(starting_point='LP2', c='qif3', ICP=[1, 11, 3],
                                                            NMX=12000, DSMAX=0.05, origin=eta_hb2_cont,
                                                            bidirectional=True, name='eta_alpha_lp3', NDIM=n_dim,
                                                            UZR={3: alphas}, STOP={})

        if period_mapping:

            # extract limit cycle periods in eta-alpha plane
            etas = np.round(np.linspace(-6.5, -2.5, 100), decimals=4).tolist()
            period_solutions = np.zeros((len(alphas), len(etas)))
            for s, s_info in eta_alpha_lp3_solutions.items():
                if np.round(s_info['PAR(3)'], decimals=4) in alphas:
                #if 'UZ' in s_info['bifurcation']:
                    s_tmp, _ = a.run(starting_point=s, c='qif', ICP=[1, 11], UZR={1: etas}, STOP={}, EPSL=1e-6,
                                     EPSU=1e-6, EPSS=1e-4, ILP=0, ISP=0, IPS=2, DS='-', get_period=True,
                                     origin=eta_alpha_lp3_cont)
                    eta_old = etas[0]
                    for s_tmp2 in s_tmp.values():
                        if 'UZ' in s_tmp2['bifurcation'] and s_tmp2['PAR(1)'] >= eta_old:
                            idx_c = np.argwhere(np.round(s_tmp2['PAR(1)'], decimals=4) == etas)
                            idx_r = np.argwhere(np.round(s_tmp2['PAR(3)'], decimals=4) == alphas)
                            period_solutions[idx_r, idx_c] = s_tmp2['period']
                            eta_old = s_tmp2['PAR(1)']

################
# save results #
################

fname = '../results/alpha_add.pkl'
kwargs = dict()
if period_mapping:
    kwargs['period_solutions'] = period_solutions
a.to_file(fname, **kwargs)
