from pyauto import PyAuto
import numpy as np

n_dim = 4
n_params = 22

###################################
# parameter continuations in auto #
###################################

a = PyAuto("auto_files")

# initial continuation in the adaptation strength alpha
hc1_solutions, hc1_cont = a.run(e='qif_alpha_mult_hc2', c='kpr.1', NDIM=n_dim, NPAR=n_params,
                                STOP=['UZ1'], name='hc0', NMX=2000, DSMAX=0.0005, UZR={17: [0.001]})

# continuation in period of hc
hc2_solutions, hc2_cont = a.run(starting_point='UZ1', c='kpr.2', UZR={11: [0.2, 60.0]}, NDIM=n_dim, NPAR=n_params,
                                STOP=[], name='hc1', NMX=2000, origin=hc1_cont, DSMAX=0.05)

# step 3
hc3_solutions, hc3_cont = a.run(starting_point='UZ1', c='kpr.6', origin=hc2_cont, NDIM=n_dim, NPAR=n_params,
                                DSMAX=0.5, DS=0.001, DSMIN=0.0001)
