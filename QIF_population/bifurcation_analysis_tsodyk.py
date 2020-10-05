import sys
sys.path.append('../')
from pyrates.utility.pyauto import PyAuto

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
n_dim = 4
n_params = 7

###################################
# parameter continuations in auto #
###################################

a = PyAuto("auto_files")

# initial continuation in the adaptation strength alpha
alpha_0 = [0.005, 0.01, 0.02, 0.04]
alpha_solutions, alpha_cont = a.run(e='qif_tsodyks', c='qif', ICP=3, UZR={3: alpha_0},
                                    STOP=['UZ' + str(len(alpha_0))], DSMAX=0.005, NMX=4000, name='s0',
                                    NDIM=n_dim, NPAR=n_params, NPR=20)

# continue in eta for each adaptation rate alpha
solutions_eta = []
solutions_eta.append(a.run(starting_point='EP1', ICP=1, DSMAX=0.005, RL1=2.0, RL0=-12.0, NMX=4000, origin=alpha_cont,
                           bidirectional=True, name='eta_0', NDIM=n_dim, NPAR=n_params, NPR=20))

i = 1
for point, point_info in alpha_solutions.items():
    if 'UZ' in point_info['bifurcation']:
        solutions_eta.append(a.run(starting_point=point, ICP=1, DSMAX=0.005, RL1=0.0, RL0=-12.0, NMX=4000,
                                   origin=alpha_cont, bidirectional=True, name=f"eta_{i}", NDIM=n_dim, NPAR=n_params,
                                   NPR=20))
        i += 1

if codim1:

    # 1D continuations in eta for alpha > 0
    #######################################

    # choose a continuation in eta to run further continuations on
    eta_points, eta_cont = solutions_eta[4]

    # continue first hopf in eta
    eta_hb1_solutions, eta_hb1_cont = a.run(starting_point='HB1', c='qif2b', ICP=[1, 11], DSMAX=0.05, NMX=4000,
                                            origin=eta_cont, name='eta/hb1', STOP={'BP1'}, NDIM=n_dim,
                                            NPAR=n_params, NPR=20)

    # # continue second hopf in eta
    # eta_hb2_solutions, eta_hb2_cont = a.run(starting_point='HB2', c='qif2b', ICP=[1, 11], DSMAX=0.05, NMX=4000,
    #                                         origin=eta_cont, name='eta/hb2', NDIM=n_dim, NPAR=n_params, NPR=20,
    #                                         STOP={'BP1'})

    if codim2:

        # 2D continuations of hopf in eta, alpha and U0
        ###############################################

        # continue the limit cycle borders in u0 and eta
        eta_u0_hb2_solutions, eta_u0_hb2_cont = a.run(starting_point='HB2', c='qif2', ICP=[6, 1], DSMAX=0.01,
                                                      NMX=4000, bidirectional=True, origin=eta_cont, RL0=0.0,
                                                      RL1=1.0, name='u0/eta/hb2', ILP=0, NDIM=n_dim, NPAR=n_params,
                                                      NPR=20)

        # continue the limit cycle borders in alpha and eta
        eta_alpha_hb2_solutions, eta_alpha_hb2_cont = a.run(starting_point='HB2', c='qif2', ICP=[3, 1], DSMAX=0.01,
                                                            NMX=4000, bidirectional=True, origin=eta_cont, RL0=0.0,
                                                            RL1=1.0, name='alpha/eta/hb2', ILP=0, NDIM=n_dim,
                                                            NPAR=n_params, NPR=20)

        # # continue the limit cycle borders in u0 and alpha
        # u0_alpha_hb2_solutions, u0_alpha_hb2_cont = a.run(starting_point='HB2', c='qif2', ICP=[6, 3], DSMAX=0.01,
        #                                                   NMX=4000, bidirectional=True, origin=eta_cont, RL0=0.0,
        #                                                   RL1=1.0, name='u0/alpha/hb2', ILP=0, NDIM=n_dim,
        #                                                   NPAR=n_params, NPR=20)

        # 2D continuations of fold in eta, alpha and U0
        ###############################################

        # choose a continuation in eta to run further continuations on
        eta_points, eta_cont = solutions_eta[1]

        # continue the limit cycle borders in u0 and eta
        eta_u0_lp1_solutions, eta_u0_lp1_cont = a.run(starting_point='LP1', c='qif2', ICP=[6, 1], DSMAX=0.01,
                                                      NMX=4000, bidirectional=True, origin=eta_cont, RL0=0.0,
                                                      RL1=1.0, name='u0/eta/lp1', ILP=0, NDIM=n_dim, NPAR=n_params,
                                                      NPR=20)

        # continue the limit cycle borders in alpha and eta
        eta_alpha_lp1_solutions, eta_alpha_lp1_cont = a.run(starting_point='LP1', c='qif2', ICP=[3, 1], DSMAX=0.01,
                                                            NMX=4000, bidirectional=True, origin=eta_cont, RL0=0.0,
                                                            RL1=1.0, name='alpha/eta/lp1', ILP=0, NDIM=n_dim,
                                                            NPAR=n_params, NPR=20)

        # # continue the limit cycle borders in u0 and alpha
        # u0_alpha_lp1_solutions, u0_alpha_lp1_cont = a.run(starting_point='LP1', c='qif2', ICP=[6, 3], DSMAX=0.01,
        #                                                   NMX=4000, bidirectional=True, origin=eta_cont, RL0=0.0,
        #                                                   RL1=1.0, name='u0/alpha/lp1', ILP=0, NDIM=n_dim,
        #                                                   NPAR=n_params, NPR=20)

################
# save results #
################

fname = '../results/tsodyks.hdf5'
a.to_file(fname)

#if period_mapping:
#    period_solutions.tofile(f"biexp_mult_period")