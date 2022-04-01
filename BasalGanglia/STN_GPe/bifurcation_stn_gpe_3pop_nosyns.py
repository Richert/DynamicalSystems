from pyrates.utility.pyauto import PyAuto
import sys
from matplotlib.pyplot import show

"""Bifurcation analysis of STN-GPe model with two GPe populations (arkypallidal and prototypical) and 
gamma-distributed axonal delays and bi-exponential synapses."""

path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"

# config
n_dim = 7
n_params = 27
a = PyAuto("auto_files", auto_dir=auto_dir)
kwargs = dict()
model = 'stn_gpe_3pop_nosyns'
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
kp_sols, kp_cont = a.run(starting_point=starting_point, c='qif', ICP=20, NPAR=n_params, name='k_p', NDIM=n_dim,
                         RL0=0.8, RL1=2.1, origin=starting_cont, NMX=2000, DSMAX=0.1, UZR={20: [2.0]}, STOP={})

starting_point = 'UZ1'
starting_cont = kp_cont

# choose relative strength of inter- vs. intra-population coupling inside GPe
ki_sols, ki_cont = a.run(starting_point=starting_point, c='qif', ICP=21, NPAR=n_params, name='k_i', NDIM=n_dim,
                         RL0=0.8, RL1=2.1, origin=starting_cont, NMX=2000, DSMAX=0.1, UZR={21: [1.5]}, STOP={})

starting_point = 'UZ1'
starting_cont = ki_cont

# preparation of healthy state
##############################

# continuation of eta_a
c0_sols, c0_cont = a.run(starting_point=starting_point, c='qif', ICP=3, NPAR=n_params,
                         name=f'eta_a:tmp', NDIM=n_dim, RL0=-1.0, RL1=10.0, origin=starting_cont,
                         NMX=8000, DSMAX=0.1, UZR={3: 1.0})
starting_point = 'UZ1'
starting_cont = c0_cont

# continuation of eta_e
c1_sols, c1_cont = a.run(starting_point=starting_point, c='qif', ICP=1, NPAR=n_params,
                         name=f'eta_e:tmp', NDIM=n_dim, RL0=-1.0, RL1=10.0, origin=starting_cont,
                         NMX=8000, DSMAX=0.1, UZR={1: [4.0]})
starting_point = 'UZ1'
starting_cont = c1_cont

# continuation of eta_p
c2_sols, c2_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params,
                         name=f'eta_p:tmp', NDIM=n_dim, RL0=-1.0, RL1=10.0, origin=starting_cont,
                         NMX=4000, DSMAX=0.1, UZR={2: [4.0]})
starting_point = 'UZ1'
starting_cont = c2_cont

# continuation of k_ae
c3_sols, c3_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif', ICP=6, NDIM=n_dim,
                         NPAR=n_params, RL1=10.0, NMX=4000, DSMAX=0.05, name=f'k_ae:tmp',
                         UZR={6: [3.0]})

starting_point = 'UZ1'
starting_cont = c3_cont

# continuation of k_pe
c4_sols, c4_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif', ICP=5, NDIM=n_dim,
                         NPAR=n_params, RL1=25.0, NMX=4000, DSMAX=0.05, name=f'k_pe:tmp',
                         UZR={5: [8.0, 17.0]})

starting_point = 'UZ1'
starting_cont = c4_cont

# continuation of pd-related parameters
#######################################

# continuations of k_gp for k_pe = 14.0
c5_sols, c5_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif', ICP=19, NDIM=n_dim,
                         NPAR=n_params, RL0=0.0, RL1=15.0, NMX=12000, DSMAX=0.05, name=f'k_gp:1',
                         bidirectional=True, UZR={19: [5.0]})
# a.run(starting_point='HB1', origin=c5_cont, c='qif2b', ICP=[19, 11], NDIM=n_dim, NPAR=n_params, RL0=0.0, RL1=15.0,
#       NMX=2000, DSMAX=0.2, name=f'k_gp:1:lc1', STOP=['BP1'], NPR=10)

# continuations of k_gp for k_pe = 17.0
c6_sols, c6_cont = a.run(starting_point='UZ2', origin=starting_cont, c='qif', ICP=19, NDIM=n_dim,
                         NPAR=n_params, RL0=0.0, RL1=15.0, NMX=12000, DSMAX=0.05, name=f'k_gp:2', bidirectional=True)
# a.run(starting_point='HB3', origin=c6_cont, c='qif2b', ICP=[19, 11], NDIM=n_dim, NPAR=n_params, RL0=10.0, RL1=15.0,
#       NMX=2000, DSMAX=0.2, name=f'k_gp:2:lc1', STOP=['BP1'], NPR=10)
# a.run(starting_point='HB4', origin=c6_cont, c='qif2b', ICP=[19, 11], NDIM=n_dim, NPAR=n_params, RL0=10.0, RL1=15.0,
#       NMX=2000, DSMAX=0.2, name=f'k_gp:2:lc2', STOP=['BP1'], NPR=10)

# continuations of k_pe for k_gp = 5.0
c7_sols, c7_cont = a.run(starting_point='UZ1', origin=c5_cont, c='qif', ICP=5, NDIM=n_dim,
                         NPAR=n_params, RL0=0.0, RL1=25.0, NMX=16000, DSMAX=0.05, name=f'k_pe:1',
                         UZR={5: [4.0]}, DS='-')
# a.run(starting_point='HB1', origin=c7_cont, c='qif2b', ICP=[5, 11], NDIM=n_dim, NPAR=n_params, RL0=0.0, RL1=10.0,
#       NMX=2000, DSMAX=0.2, name=f'k_pe:1:lc1', STOP=['BP1'], NPR=10)
# a.run(starting_point='HB4', origin=c7_cont, c='qif2b', ICP=[5, 11], NDIM=n_dim, NPAR=n_params, RL0=0.0, RL1=10.0,
#       NMX=2000, DSMAX=0.2, name=f'k_pe:1:lc2', STOP=['BP1'], NPR=10)

# 2D continuation of k_gp and k_pe
c5_2d1_sols, c5_2d1_cont = a.run(starting_point='HB1', origin=c5_cont, c='qif2', ICP=[5, 19], NDIM=n_dim, NPAR=n_params,
                                 RL0=0.0, RL1=20.0, NMX=20000, DSMAX=0.1, name=f'k_gp/k_pe:hb1', bidirectional=True)
c7_2d1_sols, c7_2d1_cont = a.run(starting_point='LP2', origin=c7_cont, c='qif2', ICP=[5, 19], NDIM=n_dim, NPAR=n_params,
                                 RL0=0.0, RL1=20.0, NMX=20000, DSMAX=0.1, name=f'k_gp/k_pe:lp1', bidirectional=True)

# save results
a.to_file(fname, **kwargs)

# plotting
ax = a.plot_continuation('PAR(19)', 'PAR(5)', cont='k_gp/k_pe:hb1', line_style_unstable='solid', ignore=['UZ'])
# ax = a.plot_continuation('PAR(19)', 'PAR(5)', cont='k_gp/k_pe:hb2', line_style_unstable='solid', ignore=['UZ'], ax=ax,
#                          line_color_stable='#148F77', line_color_unstable='#148F77')
ax = a.plot_continuation('PAR(19)', 'PAR(5)', cont='k_gp/k_pe:lp1', line_style_unstable='solid', ignore=['UZ'], ax=ax,
                         line_color_stable='#3689c9', line_color_unstable='#3689c9')
ax.set_ylabel(r'$k_{pe}$')
ax.set_xlabel(r'$k_{gp}$')
ax.set_xlim([0, 15])
ax.set_ylim([0, 20])

show()
