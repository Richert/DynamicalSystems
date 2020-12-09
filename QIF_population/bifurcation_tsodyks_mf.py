from pyrates.utility.pyauto import PyAuto
import numpy as np
import sys

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

path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"
fname = '../results/tsodyks_mf.pkl'
kwargs = dict()

codim1 = True
codim2 = True

m = 100
n_dim = 4*m
n_params = 9

###################################
# parameter continuations in auto #
###################################

a = PyAuto("auto_files", auto_dir=auto_dir)

# initial continuation in time to converge to fixed point
t_sols, t_cont = a.run(e='qif_xu_fp', c='ivp', ICP=14, DS=5e-3, DSMIN=1e-4, DSMAX=1.0, NMX=1000000, name='t',
                       UZR={14: 1000.0}, STOP={'UZ1'}, NDIM=n_dim, NPAR=n_params)

# 1D continuation in delta
delta_sols, delta_cont = a.run(starting_point='UZ1', c='qifa', ICP=5, DSMAX=0.005, RL0=0.0001, RL1=2.0, NMX=8000,
                               NPAR=n_params, origin=t_cont, name=f"delta", NDIM=n_dim, NPR=20, DS=1e-3,
                               UZR={5: [0.4]}, STOP=['UZ1'])

# 1D continuation in alpha
alpha_sols, alpha_cont = a.run(starting_point='UZ1', c='qifa', ICP=8, DSMAX=0.05, RL0=0.0, RL1=1.0, NMX=8000,
                               NPAR=n_params, origin=delta_cont, name=f"alpha", NDIM=n_dim, NPR=20, DS=1e-3,
                               UZR={8: [0.04]}, STOP=['UZ1'])
alpha2_sols, alpha2_cont = a.run(starting_point='UZ1', c='qifa', ICP=8, DSMAX=0.05, RL0=0.0, RL1=1.0, NMX=8000,
                                 NPAR=n_params, origin=t_cont, name=f"alpha2", NDIM=n_dim, NPR=20, DS=1e-3,
                                 UZR={8: [0.04]}, STOP=['UZ1'])

# 1D continuation in U0
u0_sols, u0_cont = a.run(starting_point='UZ1', c='qifa', ICP=9, DSMAX=0.05, RL0=0.0, RL1=1.0, NMX=8000,
                         NPAR=n_params, origin=delta_cont, name=f"u0", NDIM=n_dim, NPR=20, DS='-',
                         UZR={9: [0.2]}, STOP=['UZ1'])
u02_sols, u02_cont = a.run(starting_point='UZ1', c='qifa', ICP=9, DSMAX=0.05, RL0=0.0, RL1=1.0, NMX=8000,
                           NPAR=n_params, origin=t_cont, name=f"u02", NDIM=n_dim, NPR=20, DS='-',
                           UZR={9: [0.2]}, STOP=['UZ1'])

if codim1:

    # principle continuation in eta
    ###############################

    # 1D continuation in eta for delta = 0.4, alpha = 0.04 and U0 = 1.0
    eta_sols, eta_cont = a.run(starting_point='UZ1', c='qifa', ICP=1, DSMAX=0.01, RL0=-5.0, RL1=2.0, NMX=8000,
                               origin=alpha_cont, name=f"eta", NDIM=n_dim, NPAR=n_params, NPR=20, DSMIN=1e-9)

    # 1D continuation in eta for delta = 0.01, alpha = 0.04 and U0 = 1.0
    eta2_sols, eta2_cont = a.run(starting_point='UZ1', c='qifa', ICP=1, DSMAX=0.01, RL0=-5.0, RL1=2.0, NMX=8000,
                                 origin=alpha2_cont, name=f"eta_2", NDIM=n_dim, NPAR=n_params, NPR=20, DSMIN=1e-9)

    # 1D continuation in eta for delta = 0.4, alpha = 0.0 and U0 = 0.2
    eta3_sols, eta3_cont = a.run(starting_point='UZ1', c='qifa', ICP=1, DSMAX=0.01, RL0=-5.0, RL1=2.0, NMX=8000,
                                 origin=u0_cont, name=f"eta_3", NDIM=n_dim, NPAR=n_params, NPR=20, DSMIN=1e-9)

    # 1D continuation in eta for delta = 0.01, alpha = 0.0 and U0 = 0.2
    eta4_sols, eta4_cont = a.run(starting_point='UZ1', c='qifa', ICP=1, DSMAX=0.01, RL0=-5.0, RL1=2.0, NMX=8000,
                                 origin=u02_cont, name=f"eta_4", NDIM=n_dim, NPAR=n_params, NPR=20, DSMIN=1e-9)

    a.to_file(fname, **kwargs)

    if codim2:

        # 2D continuations of hopf and fold curves
        ##########################################

        # continue the limit cycle borders in Delta and eta
        delta_eta_hb2_solutions, delta_eta_hb2_cont = a.run(starting_point='HB2', c='qif2', ICP=[5, 1], DSMAX=0.05,
                                                            NMX=2000, origin=eta_cont, name='eta_Delta_hb1',
                                                            NDIM=n_dim, NPAR=n_params, RL0=0.001, RL1=1.0,
                                                            ILP=0, NPR=10, bidirectional=True, DSMIN=1e-5)
        a.to_file(fname, **kwargs)

        # continue the fold borders in Delta and eta
        delta_eta_lp1_solutions, delta_eta_lp1_cont = a.run(starting_point='LP1', c='qif2', ICP=[5, 1], DSMAX=0.05,
                                                            NMX=2000, origin=eta_cont, name='eta_Delta_lp1',
                                                            NDIM=n_dim, NPAR=n_params, RL0=0.001, RL1=1.0,
                                                            ILP=0, NPR=10, bidirectional=True, DSMIN=1e-5)
        a.to_file(fname, **kwargs)

        delta_eta_lp2_solutions, delta_eta_lp2_cont = a.run(starting_point='LP1', c='qif2', ICP=[5, 1], DSMAX=0.05,
                                                            NMX=2000, origin=eta3_cont, name='eta_Delta_lp2',
                                                            NDIM=n_dim, NPAR=n_params, RL0=0.001, RL1=1.0,
                                                            ILP=0, NPR=10, bidirectional=True, DSMIN=1e-5)
        a.to_file(fname, **kwargs)

    # continuation of selected limit cycles
    ########################################

    # lc continuation in eta for delta = 0.4, alpha = 0.04 and U0 = 1.0
    a.run(starting_point='HB2', c='qif2b', ICP=[1, 11], DSMAX=0.1, RL0=-5.0, RL1=2.0, NMX=2000,
          origin=eta_cont, name=f"eta:lc", NDIM=n_dim, NPAR=n_params, NPR=10, DSMIN=1e-7)
