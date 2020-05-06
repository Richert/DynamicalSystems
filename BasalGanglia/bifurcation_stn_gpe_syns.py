from pyauto import PyAuto
import numpy as np
import matplotlib.pyplot as plt

#####################################
# bifurcation analysis of qif model #
#####################################

# config
n_dim = 37
n_params = 20
a = PyAuto("auto_files")
fname = '../results/stn_gpe_syns.pkl'

# initial continuation in time
##############################

t_sols, t_cont = a.run(e='stn_gpe_syns', c='ivp', ICP=14, DS=5e-3, DSMIN=1e-4, DSMAX=1.0, NMX=1000000, name='t',
                       UZR={14: 10000.0}, STOP={'UZ1'}, NDIM=n_dim, NPAR=n_params)

# continuation in dopamine depletion (downscaling of deltas in GPe and upscaling of GPe to GPe-a coupling)
##########################################################################################################

starting_point = 'UZ1'
starting_cont = t_cont

# step 1: codim 1 investigation
c0_sols, c0_cont = a.run(starting_point=starting_point, c='qif', ICP=1, NPAR=n_params, name='dopa', NDIM=n_dim,
                         RL0=0.99, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.05)

# step 2: codim 2 investigation of branch point found in step 1
c0_2d1_sols, c0_2d1_cont = a.run(starting_point='HB1', c='qif2', ICP=[16, 1], NMX=4000, DSMAX=0.1,
                                 NPAR=n_params, name='dopa/delta_p', origin=c0_cont, NDIM=n_dim,
                                 bidirectional=True, RL0=0.01, RL1=1.0)
c0_2d2_sols, c0_2d2_cont = a.run(starting_point='HB1', c='qif2', ICP=[17, 1], NMX=4000, DSMAX=0.1,
                                 NPAR=n_params, name='dopa/delta_a', origin=c0_cont, NDIM=n_dim,
                                 bidirectional=True, RL0=0.01, RL1=1.0)
c0_2d3_sols, c0_2d3_cont = a.run(starting_point='HB1', c='qif2', ICP=[8, 1], NMX=4000, DSMAX=0.5,
                                 NPAR=n_params, name='dopa/k_ap', origin=c0_cont, NDIM=n_dim,
                                 bidirectional=True, RL0=1.0, RL1=200.0)
c0_2d4_sols, c0_2d4_cont = a.run(starting_point='HB1', c='qif2', ICP=[5, 1], NMX=4000, DSMAX=0.5,
                                 NPAR=n_params, name='dopa/k_pa', origin=c0_cont, NDIM=n_dim,
                                 bidirectional=True, RL0=1.0, RL1=200.0)

# step 3: codim 1 investigation of periodic orbit found in step 1
c0_lc1_sols, c0_lc1_cont = a.run(starting_point='HB1', c='qif2b', ICP=[1, 11], NMX=4000, DSMAX=0.5,
                                 NPAR=n_params, name='dopa_lc', origin=c0_cont, NDIM=n_dim,
                                 RL0=0.99, RL1=10.0, STOP={'PD1', 'BP2'})

# save results
kwargs = dict()
a.to_file(fname, **kwargs)

# continuation in delta_p
#########################

starting_point = 'UZ1'
starting_cont = t_cont

# step 1: codim 1 investigation
c1_sols, c1_cont = a.run(starting_point=starting_point, c='qif', ICP=16, NPAR=n_params, name='delta_p', NDIM=n_dim,
                         RL0=0.01, RL1=1.0, origin=starting_cont, NMX=6000, DSMAX=0.05, STOP={'UZ6'},
                         bidirectional=True)

# step 3: codim 1 investigation of periodic orbit found in step 1
c1_lc1_sols, c1_lc1_cont = a.run(starting_point='HB1', c='qif2b', ICP=[16, 11], NMX=4000, DSMAX=0.5,
                                 NPAR=n_params, name='delta_p_lc', origin=c1_cont, NDIM=n_dim,
                                 RL0=0.01, RL1=1.0, STOP={'PD1', 'BP2'})

# save results
kwargs = dict()
a.to_file(fname, **kwargs)

# continuation in delta_a
#########################

starting_point = 'UZ1'
starting_cont = t_cont

# step 1: codim 1 investigation
c2_sols, c2_cont = a.run(starting_point=starting_point, c='qif', ICP=17, NPAR=n_params, name='delta_a', NDIM=n_dim,
                         RL0=0.01, RL1=1.0, origin=starting_cont, NMX=6000, DSMAX=0.05, STOP={'UZ6'},
                         bidirectional=True)

# save results
kwargs = dict()
a.to_file(fname, **kwargs)

# continuation in delta_e
#########################

starting_point = 'UZ1'
starting_cont = t_cont

# step 1: codim 1 investigation
c3_sols, c3_cont = a.run(starting_point=starting_point, c='qif', ICP=15, NPAR=n_params, name='delta_e', NDIM=n_dim,
                         RL0=0.01, RL1=1.0, origin=starting_cont, NMX=6000, DSMAX=0.05, STOP={'UZ6'},
                         bidirectional=True)

# save results
kwargs = dict()
a.to_file(fname, **kwargs)

# continuation in k_ae
######################

starting_point = 'UZ1'
starting_cont = t_cont

# step 1: codim 1 investigation
c4_sols, c4_cont = a.run(starting_point=starting_point, c='qif', ICP=7, NPAR=n_params, name='k_ae', NDIM=n_dim,
                         RL0=0.0, RL1=200.0, origin=starting_cont, NMX=6000, DSMAX=0.5, STOP={},
                         bidirectional=True)

# step 3: codim 1 investigation of periodic orbit found in step 1
c4_lc1_sols, c4_lc1_cont = a.run(starting_point='HB2', c='qif2b', ICP=[7, 11], NMX=4000, DSMAX=0.5,
                                 NPAR=n_params, name='k_ae_lc', origin=c4_cont, NDIM=n_dim,
                                 RL0=1.0, RL1=200.0, STOP={'PD1', 'BP2'})

# save results
kwargs = dict()
a.to_file(fname, **kwargs)

# continuation in k_p (scaling of all synaptic inputs to prototypical population)
#################################################################################

starting_point = 'UZ1'
starting_cont = t_cont

# step 1: codim 1 investigation
c5_sols, c5_cont = a.run(starting_point=starting_point, c='qif', ICP=25, NPAR=n_params, name='k_p', NDIM=n_dim,
                         RL0=0.0, RL1=5.5, origin=starting_cont, NMX=6000, DSMAX=0.05, STOP={},
                         bidirectional=True)

# step 3: codim 1 investigation of periodic orbit found in step 1
c5_lc1_sols, c5_lc1_cont = a.run(starting_point='HB1', c='qif2b', ICP=[25, 11], NMX=4000, DSMAX=0.5,
                                 NPAR=n_params, name='k_p_lc', origin=c5_cont, NDIM=n_dim,
                                 RL0=0.0, RL1=5.5, STOP={'PD1', 'BP2'})

# save results
kwargs = dict()
a.to_file(fname, **kwargs)

# continuation in k_p (scaling of all synaptic inputs to prototypical population)
#################################################################################

starting_point = 'UZ1'
starting_cont = t_cont

# step 1: codim 1 investigation
c6_sols, c6_cont = a.run(starting_point=starting_point, c='qif', ICP=26, NPAR=n_params, name='k_gp', NDIM=n_dim,
                         RL0=0.0, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.05, STOP={},
                         bidirectional=True)

# step 3: codim 1 investigation of periodic orbit found in step 1
c6_lc1_sols, c6_lc1_cont = a.run(starting_point='HB1', c='qif2b', ICP=[26, 11], NMX=4000, DSMAX=0.5,
                                 NPAR=n_params, name='k_gp_lc', origin=c6_cont, NDIM=n_dim,
                                 RL0=0.0, RL1=10.0, STOP={'PD1', 'BP2', 'TR2'})
c6_lc2_sols, c6_lc2_cont = a.run(starting_point='HB2', c='qif2b', ICP=[26, 11], NMX=4000, DSMAX=0.5,
                                 NPAR=n_params, name='k_gp_lc2', origin=c6_cont, NDIM=n_dim,
                                 RL0=0.0, RL1=10.0, STOP={'PD1', 'BP2', 'TR2'})

# save results
kwargs = dict()
a.to_file(fname, **kwargs)

# continuation in eta_e
#######################

starting_point = 'UZ1'
starting_cont = t_cont

# step 1: codim 1 investigation
c7_sols, c7_cont = a.run(starting_point=starting_point, c='qif', ICP=18, NPAR=n_params, name='eta_e', NDIM=n_dim,
                         RL0=-10.0, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.05, STOP={},
                         bidirectional=True)

# step 2: codim 2 investigation of branch point found in step 1
c7_2d1_sols, c7_2d1_cont = a.run(starting_point='HB1', c='qif2', ICP=[2, 18], NMX=4000, DSMAX=0.1,
                                 NPAR=n_params, name='eta_e/k_ep', origin=c7_cont, NDIM=n_dim,
                                 bidirectional=True, RL0=10.0, RL1=100.0)
c7_2d2_sols, c7_2d2_cont = a.run(starting_point='HB1', c='qif2', ICP=[16, 18], NMX=4000, DSMAX=0.1,
                                 NPAR=n_params, name='eta_e/delta_p', origin=c7_cont, NDIM=n_dim,
                                 bidirectional=True, RL0=0.01, RL1=0.4)
c7_2d3_sols, c7_2d3_cont = a.run(starting_point='HB1', c='qif2', ICP=[19, 18], NMX=4000, DSMAX=0.1,
                                 NPAR=n_params, name='eta_e/eta_p', origin=c7_cont, NDIM=n_dim,
                                 bidirectional=True, RL0=-10.0, RL1=10.0)
c7_2d4_sols, c7_2d4_cont = a.run(starting_point='HB1', c='qif2', ICP=[20, 18], NMX=4000, DSMAX=0.1,
                                 NPAR=n_params, name='eta_e/eta_a', origin=c7_cont, NDIM=n_dim,
                                 bidirectional=True, RL0=-10.0, RL1=10.0)

# step 3: codim 1 investigation of periodic orbit found in step 1
c7_lc1_sols, c7_lc1_cont = a.run(starting_point='HB1', c='qif2b', ICP=[18, 11], NMX=4000, DSMAX=0.5,
                                 NPAR=n_params, name='eta_e_lc', origin=c7_cont, NDIM=n_dim,
                                 RL0=-10.0, RL1=10.0, STOP={'PD1', 'BP2'})

# save results
kwargs = dict()
a.to_file(fname, **kwargs)