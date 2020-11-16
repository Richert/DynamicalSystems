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

codim1 = True
codim2 = False
n_grid_points = 100
m = 100
n_dim = 3*m
n_params = 9
eta_cont_idx = 0

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
                         NPAR=n_params, origin=delta_cont, name=f"alpha", NDIM=n_dim, NPR=20, DS='-',
                         UZR={9: [0.2]}, STOP=['UZ1'])

# principle continuation in eta
###############################

# 1D continuation in eta for delta = 0.4, alpha = 0.04 and U0 = 1.0
eta_sols, eta_cont = a.run(starting_point='UZ1', c='qifa', ICP=1, DSMAX=0.005, RL0=-5.0, RL1=5.0, NMX=8000,
                           origin=alpha_cont, name=f"eta", NDIM=n_dim, NPAR=n_params, NPR=20, DS=1e-3)

# 1D continuation in eta for delta = 0.01, alpha = 0.04 and U0 = 1.0
eta2_sols, eta2_cont = a.run(starting_point='UZ1', c='qifa', ICP=1, DSMAX=0.005, RL0=-5.0, RL1=5.0, NMX=8000,
                             origin=alpha2_cont, name=f"eta_2", NDIM=n_dim, NPAR=n_params, NPR=20, DS=1e-3)

# 1D continuation in eta for delta = 0.4, alpha = 0.0 and U0 = 0.2
eta_sols3, eta_cont3 = a.run(starting_point='UZ1', c='qifa', ICP=1, DSMAX=0.005, RL0=-5.0, RL1=5.0, NMX=8000,
                             origin=u0_cont, name=f"eta_3", NDIM=n_dim, NPAR=n_params, NPR=20, DS=1e-3)

################
# save results #
################

fname = '../results/tsodyks_mf.pkl'
kwargs = dict()
a.to_file(fname, **kwargs)
