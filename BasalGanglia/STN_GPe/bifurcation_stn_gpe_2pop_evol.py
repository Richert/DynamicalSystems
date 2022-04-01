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
model = 'stn_gpe_2pop_evol'
fname = f'../results/{model}_conts.pkl'

#########################
# initial continuations #
#########################

# continuation in time
t_sols, t_cont = a.run(e=model, c='ivp', ICP=14, NMX=1000000, name='t', UZR={14: 1000.0}, STOP={'UZ1'},
                       NDIM=n_dim, NPAR=n_params)

starting_point = 'UZ1'
starting_cont = t_cont

# preparation of oscillatory state
##################################

# continuation of eta_p: v1 = -5.0, -3.26
c1_sols, c1_cont = a.run(starting_point=starting_point, c='qif', ICP=16, NPAR=n_params,
                         name=f'eta_p:tmp', NDIM=n_dim, RL1=10.0, origin=starting_cont,
                         NMX=4000, DSMAX=0.1, UZR={16: [-3.26]}, STOP=['UZ1'])

starting_point = 'UZ1'
starting_cont = c1_cont

# continuation of eta_e: v1 = 4.83, v2 = 6.69
c2_sols, c2_cont = a.run(starting_point=starting_point, c='qif', ICP=15, NPAR=n_params,
                         name=f'eta_e:tmp', NDIM=n_dim, RL1=10.0, origin=starting_cont,
                         NMX=8000, DSMAX=0.1, UZR={15: [6.69]}, STOP=['UZ1'])

starting_point = 'UZ1'
starting_cont = c2_cont

# continuation of pd-related parameters
#######################################

# continuations of k_gp for k_pe = 16.0
c3_sols, c3_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif', ICP=20, NDIM=n_dim,
                         NPAR=n_params, RL0=0.0, RL1=20.0, NMX=12000, DSMAX=0.05, name=f'k_gp:1',
                         bidirectional=True, UZR={20: [5.0]})
a.run(starting_point='HB1', origin=c3_cont, c='qif2b', ICP=[20, 11], NDIM=n_dim, NPAR=n_params, RL0=0.0, RL1=15.0,
      NMX=2000, DSMAX=0.2, name=f'k_gp:1:lc1', STOP=['BP1'], NPR=10)

# 2D continuation of k_gp and k_pe
c3_2d1_sols, c3_2d1_cont = a.run(starting_point='HB1', origin=c3_cont, c='qif2', ICP=[17, 20], NDIM=n_dim,
                                 NPAR=n_params, RL0=0.0, RL1=25.0, NMX=20000, DSMAX=0.1,
                                 name=f'k_gp/k_pe:hb1', bidirectional=True)

# plotting
import matplotlib.pyplot as plt

fig, axes = plt.subplots(ncols=2)

ax1 = axes[0]
ax1 = a.plot_continuation('PAR(20)', 'PAR(17)', cont='k_gp/k_pe:hb1', line_style_unstable='solid', ignore=['UZ'], ax=ax1)
ax1.set_xlabel(r'$k_{gp}$')
ax1.set_ylabel(r'$k_{pe}$')

ax2 = axes[1]
ax2 = a.plot_continuation('PAR(20)', 'U(3)', cont='k_gp:1', line_style_unstable='solid', ignore=['UZ'], ax=ax2)
ax2 = a.plot_continuation('PAR(20)', 'U(3)', cont='k_gp:1:lc1', line_style_unstable='solid', ignore=['UZ'], ax=ax2)
ax2.set_xlabel(r'$k_{gp}$')
ax2.set_ylabel(r'$r$')

plt.show()
