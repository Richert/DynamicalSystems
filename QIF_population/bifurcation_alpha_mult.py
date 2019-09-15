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
n_grid_points = 100
n_dim = 4
n_params = 6
eta_cont_idx = 3

###################################
# parameter continuations in auto #
###################################

a = PyAuto("auto_files")

# initial continuation in the adaptation strength alpha
alpha_0 = [0.0125, 0.025, 0.05, 0.1, 0.2]
alpha_solutions, alpha_cont = a.run(e='qif_alpha_mult', c='qif', ICP=3, UZR={3: alpha_0}, NDIM=n_dim,
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
    eta_hb2_solutions, eta_hb2_cont = a.run(starting_point='HB2', c='qif2b', ICP=[1, 11], DSMAX=0.05, NMX=4000,
                                            origin=eta_cont, name='eta_hb2', NDIM=n_dim)

    # continuation in eta and alpha
    ###############################

    if codim2:

        # continue the limit cycle borders in alpha and eta
        eta_alpha_hb2_solutions, eta_alpha_hb2_cont = a.run(starting_point='HB2', c='qif2', ICP=[1, 3], DSMAX=0.01,
                                                            NMX=2000, bidirectional=True, origin=eta_cont,
                                                            name='eta_alpha_hb2', NDIM=n_dim)

        # continue the first folding bifurcation of the limit cycle
        alphas = np.round(np.linspace(0.02, 0.16, 100)[::-1], decimals=4).tolist()
        eta_alpha_pd_solutions, eta_alpha_pd_cont = a.run(starting_point='LP1', c='qif3', ICP=[1, 11, 3],
                                                          NMX=4000, DSMAX=0.05, origin=eta_hb2_cont,
                                                          bidirectional=True, name='eta_alpha_lp', NDIM=n_dim,
                                                          UZR={3: alphas}, STOP={})

################
# save results #
################

fname = '../results/alpha_mult.pkl'
a.to_file(fname)
