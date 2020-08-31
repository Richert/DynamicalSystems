import sys
sys.path.append('../')
import os
from pyrates.utility.pyauto import PyAuto
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
n_params = 6

###################################
# parameter continuations in auto #
###################################

a = PyAuto("auto_files")

# initial continuation in the adaptation strength alpha
alpha_0 = [0.0125, 0.025, 0.05, 0.1, 0.2]
alpha_solutions, alpha_cont = a.run(e='qif_biexp_mult', c='qif', ICP=3, UZR={3: alpha_0},
                                    STOP=['UZ' + str(len(alpha_0))], DSMAX=0.005, NMX=4000, name='s0',
                                    NDIM=n_dim, NPAR=n_params)

'''
# continue in tau_r for all values of alpha
solutions_tau = []
solutions_tau.append(a.run(starting_point='EP1', ICP=4, DSMAX=0.001, RL1=10.01, RL0=0.001, NMX=10000, origin=alpha_cont,
                           bidirectional=True, name='tau_0', NDIM=n_dim, NPAR=n_params))

i = 1
for point, point_info in alpha_solutions.items():
    if 'UZ' in point_info['bifurcation']:
        solutions_tau.append(a.run(starting_point=point, ICP=4, DSMAX=0.001, RL1=10.01, RL0=0.001, NMX=10000,
                                   origin=alpha_cont, bidirectional=True, name=f"tau_{i}", NDIM=n_dim, NPAR=n_params))
        i += 1


# choose a continuation in tau to run further continuations on (alpha = 0.05)
tau_points, tau_cont = solutions_tau[3]
'''

'''
# principle continuation in eta
###############################
# continue in eta for each adaptation rate alpha
solutions_eta = []
solutions_eta.append(a.run(starting_point='EP1', ICP=1, DSMAX=0.005, RL1=0.0, RL0=-12.0, NMX=2000, origin=alpha_cont,#tau_cont
                           bidirectional=True, name='eta_0', NDIM=n_dim, NPAR=n_params))

i = 1
for point, point_info in alpha_solutions.items():
    if 'UZ' in point_info['bifurcation']:
        solutions_eta.append(a.run(starting_point=point, ICP=1, DSMAX=0.005, RL1=0.0, RL0=-12.0, NMX=4000,
                                   origin=alpha_cont, bidirectional=True, name=f"eta_{i}", NDIM=n_dim, NPAR=n_params))#tau_cont
        i += 1



# choose a continuation in eta to run further continuations on
eta_points, eta_cont = solutions_eta[3]
'''


tau_points, tau_cont = a.run(starting_point='UZ3', ICP=4, DSMAX=0.001, RL1=11, RL0=0.001, NMX=100000, origin=alpha_cont,
                           bidirectional=True, name='tau_0', NDIM=n_dim, NPAR=n_params, UZR={4:[0.3]})

# continuation for small tau
eta_points, eta_cont = a.run(starting_point='UZ1', ICP=1, DSMAX=0.001, RL1=0.0, RL0=-12.0, NMX=100000,
                                   origin=tau_cont, bidirectional=True, name="eta_smalltau", NDIM=n_dim, NPAR=n_params)


if codim1:

    # 1D continuations in eta and tau1
    ##################################

    # limit cycle continuation of hopf bifurcations in eta
    eta_hb1_solutions, eta_hb1_cont = a.run(starting_point='HB1', c='qif2b', ICP=[1, 11], DSMAX=0.01, NMX=10000,
                                            origin=eta_cont, name='eta/hb1', STOP={'BP1'}, NDIM=n_dim, NPAR=n_params)
    eta_hb2_solutions, eta_hb2_cont = a.run(starting_point='HB2', c='qif2b', ICP=[1, 11], DSMAX=0.01, NMX=10000,
                                            origin=eta_cont, name='eta/hb2', UZR={1: [-5.22]}, NDIM=n_dim, NPAR=n_params)



################
# save results #
################

fname = '../results/biexp_mult_smalltau.hdf5'
a.to_file(fname)

#if period_mapping:
#    period_solutions.tofile(f"biexp_mult_period")
