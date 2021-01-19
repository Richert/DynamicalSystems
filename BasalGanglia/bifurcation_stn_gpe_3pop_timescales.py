from pyrates.utility.pyauto import PyAuto
import sys
from matplotlib.pyplot import show

"""Bifurcation analysis of STN-GPe model with two GPe populations (arkypallidal and prototypical) and 
gamma-distributed axonal delays and bi-exponential synapses."""

path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"

# config
n_dim = 19
n_params = 32
a = PyAuto("auto_files", auto_dir=auto_dir)
kwargs = dict()
model = 'stn_gpe_3pop_timescales'
fname = f'../results/{model}_conts.pkl'

#########################
# initial continuations #
#########################

# continuation in time
t_sols, t_cont = a.run(e=model, c='ivp', ICP=14, NMX=100000, name='t', UZR={14: 1000.0}, STOP={'UZ1'},
                       NDIM=n_dim, NPAR=n_params)
starting_point = 'UZ1'
starting_cont = t_cont

# choose relative strength of GPe-p vs. GPe-a axons
kp_sols, kp_cont = a.run(starting_point=starting_point, c='qif', ICP=23, NPAR=n_params, name='k_p', NDIM=n_dim,
                         RL0=0.8, RL1=2.1, origin=starting_cont, NMX=2000, DSMAX=0.1, UZR={23: [2.0]}, STOP={})

# choose relative strength of inter- vs. intra-population coupling inside GPe
ki_sols, ki_cont = a.run(starting_point=starting_point, c='qif', ICP=24, NPAR=n_params, name='k_i', NDIM=n_dim,
                         RL0=0.8, RL1=2.1, origin=starting_cont, NMX=2000, DSMAX=0.1, UZR={24: [1.5]}, STOP={})

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

# continuation of k_ae
c2_b3_sols, c2_b3_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif', ICP=6, NDIM=n_dim,
                               NPAR=n_params, RL1=10.0, NMX=4000, DSMAX=0.05, name=f'k_ae:tmp',
                               UZR={6: [5.0]})

# continuation of k_stn
c2_b4_sols, c2_b4_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif', ICP=25, NDIM=n_dim,
                               NPAR=n_params, RL1=10.0, NMX=4000, DSMAX=0.05, name=f'k_stn:tmp',
                               UZR={25: [2.0]})

# continuation of pd-related parameters
#######################################

# continuations of k_gp for k_stn = 2.0
c2_b5_sols, c2_b5_cont = a.run(starting_point='UZ1', origin=c2_b4_cont, c='qif', ICP=22, NDIM=n_dim,
                               NPAR=n_params, RL0=0.0, RL1=20.0, NMX=12000, DSMAX=0.1, name=f'k_gp:1',
                               bidirectional=True)

# 2D Hopf curve: tau_e x tau_p
c2_b5_2d1_sols, c2_b5_2d1_cont = a.run(starting_point='HB1', origin=c2_b5_cont, c='qif2', ICP=[26, 27], NDIM=n_dim,
                                       NPAR=n_params, RL0=5.0, RL1=15.0, NMX=8000, DSMAX=0.1,
                                       name=f'tau_e/tau_p:hb1', bidirectional=True)

# 2D Hopf curve: tau_p x tau_a
c2_b5_2d2_sols, c2_b5_2d2_cont = a.run(starting_point='HB1', origin=c2_b5_cont, c='qif2', ICP=[27, 28], NDIM=n_dim,
                                       NPAR=n_params, RL0=10.0, RL1=30.0, NMX=8000, DSMAX=0.1,
                                       name=f'tau_p/tau_a:lc:lp1', bidirectional=True)

# 2D Hopf curve: tau_e x tau_a
c2_b5_2d3_sols, c2_b5_2d3_cont = a.run(starting_point='HB1', origin=c2_b5_cont, c='qif2', ICP=[26, 28], NDIM=n_dim,
                                       NPAR=n_params, RL0=5.0, RL1=15.0, NMX=8000, DSMAX=0.1,
                                       name=f'tau_e/tau_a:hb1', bidirectional=True)

# 2D Hopf curve: tau_ampa_d x tau_gabaa_d
c2_b5_2d4_sols, c2_b5_2d4_cont = a.run(starting_point='HB1', origin=c2_b5_cont, c='qif2', ICP=[30, 32], NDIM=n_dim,
                                       NPAR=n_params, RL0=2.0, RL1=8.0, NMX=8000, DSMAX=0.1,
                                       name=f'tau_ampa_d/tau_gabaa_d:hb1', bidirectional=True)

# 2D Hopf curve: delta_e x delta_p
c2_b5_2d5_sols, c2_b5_2d5_cont = a.run(starting_point='HB1', origin=c2_b5_cont, c='qif2', ICP=[18, 19], NDIM=n_dim,
                                       NPAR=n_params, RL0=1.0, RL1=6.0, NMX=10000, DSMAX=0.1,
                                       name=f'delta_e/delta_p:hb1', bidirectional=True)

# 1D lc continuation in tau_e
a.run(starting_point='HB1', origin=c2_b5_cont, c='qif2b', ICP=[26, 11], NDIM=n_dim, NPAR=n_params, RL0=6.0, RL1=20.0,
      NMX=2000, DSMAX=0.1, name=f'tau_e:lc1', STOP=['BP1'], NPR=10)

# 1D lc continuation in tau_p
a.run(starting_point='HB1', origin=c2_b5_cont, c='qif2b', ICP=[27, 11], NDIM=n_dim, NPAR=n_params, RL0=10.0, RL1=30.0,
      NMX=2000, DSMAX=0.1, name=f'tau_p:lc1', STOP=['BP1'], NPR=10)

# 1D lc continuation in tau_a
a.run(starting_point='HB1', origin=c2_b5_cont, c='qif2b', ICP=[28, 11], NDIM=n_dim, NPAR=n_params, RL0=10.0, RL1=30.0,
      NMX=2000, DSMAX=0.1, name=f'tau_a:lc1', STOP=['BP1'], NPR=10)

# 1D lc continuation in tau_ampa_d
a.run(starting_point='HB1', origin=c2_b5_cont, c='qif2b', ICP=[30, 11], NDIM=n_dim, NPAR=n_params, RL0=3.0, RL1=8.0,
      NMX=2000, DSMAX=0.1, name=f'tau_ampa_d:lc1', STOP=['BP1'], NPR=10)

# 1D lc continuation in tau_gabaa_d
a.run(starting_point='HB1', origin=c2_b5_cont, c='qif2b', ICP=[32, 11], NDIM=n_dim, NPAR=n_params, RL0=4.0, RL1=10.0,
      NMX=2000, DSMAX=0.1, name=f'tau_gabaa_d:lc1', STOP=['BP1'], NPR=10)

# 1D lc continuation in delta_e
a.run(starting_point='HB1', origin=c2_b5_cont, c='qif2b', ICP=[18, 11], NDIM=n_dim, NPAR=n_params, RL0=1.0, RL1=6.0,
      NMX=2000, DSMAX=0.1, name=f'delta_e:lc1', STOP=['BP1'], NPR=10)

# 1D lc continuation in delta_p
a.run(starting_point='HB1', origin=c2_b5_cont, c='qif2b', ICP=[19, 11], NDIM=n_dim, NPAR=n_params, RL0=6.0, RL1=15.0,
      NMX=2000, DSMAX=0.1, name=f'delta_p:lc1', STOP=['BP1'], NPR=10)

# 1D lc continuation in delta_a
a.run(starting_point='HB1', origin=c2_b5_cont, c='qif2b', ICP=[20, 11], NDIM=n_dim, NPAR=n_params, RL0=10.0, RL1=20.0,
      NMX=2000, DSMAX=0.1, name=f'delta_a:lc1', STOP=['BP1'], NPR=10)

# save results
a.to_file(fname, **kwargs)

# plotting
import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=2, ncols=2)

ax = axes[0, 0]
ax = a.plot_continuation('PAR(26)', 'PAR(27)', cont='tau_e/tau_p:hb1', line_style_unstable='solid', ignore=['UZ'],
                         ax=ax)
ax.set_xlabel(r'$\tau_e$')
ax.set_ylabel(r'$\tau_p$')

ax = axes[0, 1]
ax = a.plot_continuation('PAR(27)', 'PAR(28)', cont='tau_p/tau_a:hb1', line_style_unstable='solid', ignore=['UZ'],
                         ax=ax)
ax.set_xlabel(r'$\tau_p$')
ax.set_ylabel(r'$\tau_a$')

ax = axes[1, 0]
ax = a.plot_continuation('PAR(30)', 'PAR(32)', cont='tau_ampa_d/tau_gabaa_d:hb1', line_style_unstable='solid',
                         ignore=['UZ'], ax=ax)
ax.set_xlabel(r'$\tau_d^a$')
ax.set_xlabel(r'$\tau_d^g$')

ax = axes[1, 1]
ax = a.plot_continuation('PAR(19)', 'PAR(20)', cont='delta_p/delta_a:hb1', line_style_unstable='solid',
                         ignore=['UZ'], ax=ax)
ax.set_xlabel(r'$\Delta_p$')
ax.set_xlabel(r'$\Delta_a$')

show()
