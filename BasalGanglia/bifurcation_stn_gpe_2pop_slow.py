from pyrates.utility.pyauto import PyAuto
import sys
import numpy as np
import pandas as pd

"""Bifurcation analysis of STN-GPe model with two GPe populations (arkypallidal and prototypical) and 
gamma-distributed axonal delays and bi-exponential synapses."""

path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"

# config
n_dim = 13
n_params = 22
a = PyAuto("auto_files", auto_dir=auto_dir)
kwargs = dict()
model = 'stn_gpe_2pop_timescales'
fname = f'../results/{model}_conts.pkl'

#########################
# initial continuations #
#########################

# continuation in time
t_sols, t_cont = a.run(e=model, c='ivp', ICP=14, NMX=1000000, name='t', UZR={14: 1000.0}, STOP={'UZ1'},
                       NDIM=n_dim, NPAR=n_params)

starting_point = 'UZ1'
starting_cont = t_cont

# choose relative strength of GPe-p vs. GPe-a axons
kp_sols, kp_cont = a.run(starting_point=starting_point, c='qif', ICP=16, NPAR=n_params, name='k_p', NDIM=n_dim,
                         RL0=0.8, RL1=2.1, origin=starting_cont, NMX=2000, DSMAX=0.1, UZR={16: [2.0]}, STOP={})

starting_point = 'UZ1'
starting_cont = kp_cont

# choose relative strength of inter- vs. intra-population coupling inside GPe
ki_sols, ki_cont = a.run(starting_point=starting_point, c='qif', ICP=17, NPAR=n_params, name='k_i', NDIM=n_dim,
                         RL0=0.8, RL1=2.1, origin=starting_cont, NMX=2000, DSMAX=0.1, UZR={17: [1.5]}, STOP={})

starting_point = 'UZ1'
starting_cont = ki_cont

# preparation of healthy state
##############################

# continuation of eta_e
c2_b1_sols, c2_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=1, NPAR=n_params,
                               name=f'eta_e:tmp', NDIM=n_dim, RL0=-1.0, RL1=10.0, origin=starting_cont,
                               NMX=8000, DSMAX=0.1, UZR={1: [4.0]})

starting_point = 'UZ1'
starting_cont = c2_b1_cont

# continuation of eta_p
c2_b2_sols, c2_b2_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params,
                               name=f'eta_p:tmp', NDIM=n_dim, RL0=-1.0, RL1=10.0, origin=starting_cont,
                               NMX=4000, DSMAX=0.1, UZR={2: [4.0]})

starting_point = 'UZ1'
starting_cont = c2_b2_cont

# continuation of k_pe
c2_b3_sols, c2_b3_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif', ICP=4, NDIM=n_dim,
                               NPAR=n_params, RL1=12.0, NMX=4000, DSMAX=0.05, name=f'k_pe:tmp',
                               UZR={4: [8.0]})

starting_point = 'UZ1'
starting_cont = c2_b3_cont

# continuation of tau_e
c2_b4_sols, c2_b4_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif', ICP=19, NDIM=n_dim,
                               NPAR=n_params, RL1=21.0, NMX=4000, DSMAX=0.05, name=f'tau_e:tmp',
                               UZR={19: [20.0]}, STOP=['UZ1'])

starting_point = 'UZ1'
starting_cont = c2_b4_cont

# continuation of tau_p
c2_b5_sols, c2_b5_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif', ICP=20, NDIM=n_dim,
                               NPAR=n_params, RL1=41.0, NMX=4000, DSMAX=0.05, name=f'tau_p:tmp',
                               UZR={20: [40.0]}, STOP=['UZ1'])

starting_point = 'UZ1'
starting_cont = c2_b5_cont

# continuation of tau_ampa_d
c2_b6_sols, c2_b6_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif', ICP=21, NDIM=n_dim,
                               NPAR=n_params, RL1=35.0, NMX=4000, DSMAX=0.05, name=f'tau_ampa:tmp',
                               UZR={21: [8.0]}, STOP=['UZ1'])

starting_point = 'UZ1'
starting_cont = c2_b6_cont

# continuation of tau_gabaa_d
c2_b7_sols, c2_b7_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif', ICP=22, NDIM=n_dim,
                               NPAR=n_params, RL1=35.0, NMX=4000, DSMAX=0.05, name=f'tau_gabaa:tmp',
                               UZR={22: [22.0]}, STOP=['UZ1'])

starting_point = 'UZ1'
starting_cont = c2_b7_cont

# continuation of pd-related parameters
#######################################

# continuations of k_gp for k_pe = 16.0
c2_b8_sols, c2_b8_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif', ICP=15, NDIM=n_dim,
                               NPAR=n_params, RL0=0.0, RL1=20.0, NMX=12000, DSMAX=0.05, name=f'k_gp:1',
                               bidirectional=True, UZR={15: [5.0]})
a.run(starting_point='HB1', origin=c2_b8_cont, c='qif2b', ICP=[15, 11], NDIM=n_dim, NPAR=n_params, RL0=0.0, RL1=15.0,
      NMX=2000, DSMAX=0.2, name=f'k_gp:1:lc1', STOP=['BP1'], NPR=10)

# 2D continuation of k_gp and k_pe
c2_b8_2d1_sols, c2_b8_2d1_cont = a.run(starting_point='HB1', origin=c2_b8_cont, c='qif2', ICP=[4, 15], NDIM=n_dim,
                                       NPAR=n_params, RL0=0.0, RL1=25.0, NMX=20000, DSMAX=0.1,
                                       name=f'k_gp/k_pe:hb1', bidirectional=True)

# plotting
import matplotlib.pyplot as plt

fig, axes = plt.subplots(ncols=2)

ax1 = axes[0]
ax1 = a.plot_continuation('PAR(15)', 'PAR(4)', cont='k_gp/k_pe:hb1', line_style_unstable='solid', ignore=['UZ'], ax=ax1)
ax1.set_xlabel(r'$k_{gp}$')
ax1.set_ylabel(r'$k_{pe}$')

ax2 = axes[1]
ax2 = a.plot_continuation('PAR(15)', 'U(3)', cont='k_gp:1', line_style_unstable='solid', ignore=['UZ'], ax=ax2)
ax2 = a.plot_continuation('PAR(15)', 'U(3)', cont='k_gp:1:lc1', line_style_unstable='solid', ignore=['UZ'], ax=ax2)
ax2.set_xlabel(r'$k_{gp}$')
ax2.set_ylabel(r'$r$')

plt.show()
