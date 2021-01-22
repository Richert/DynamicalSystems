from pyrates.utility.pyauto import PyAuto
import sys
from matplotlib.pyplot import show

"""Bifurcation analysis of STN-GPe model with two GPe populations (arkypallidal and prototypical) and 
gamma-distributed axonal delays and bi-exponential synapses."""

path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"

# config
n_dim = 7
n_params = 24
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
kp_sols, kp_cont = a.run(starting_point=starting_point, c='qif', ICP=23, NPAR=n_params, name='k_p', NDIM=n_dim,
                         RL0=0.8, RL1=2.1, origin=starting_cont, NMX=2000, DSMAX=0.1, UZR={23: [2.0]}, STOP={})

starting_point = 'UZ1'
starting_cont = kp_cont

# choose relative strength of inter- vs. intra-population coupling inside GPe
ki_sols, ki_cont = a.run(starting_point=starting_point, c='qif', ICP=24, NPAR=n_params, name='k_i', NDIM=n_dim,
                         RL0=0.8, RL1=2.1, origin=starting_cont, NMX=2000, DSMAX=0.1, UZR={24: [1.5]}, STOP={})

starting_point = 'UZ1'
starting_cont = ki_cont

# preparation of healthy state
##############################

# continuation of eta_a
c2_b0_sols, c2_b0_cont = a.run(starting_point=starting_point, c='qif', ICP=3, NPAR=n_params,
                               name=f'eta_a:tmp', NDIM=n_dim, RL0=-1.0, RL1=10.0, origin=starting_cont,
                               NMX=8000, DSMAX=0.1, UZR={3: 1.0})
starting_point = 'UZ1'
starting_cont = c2_b0_cont

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
                               UZR={6: [3.0]})

starting_point = 'UZ1'
starting_cont = c2_b3_cont

# continuation of k_pe
c2_b4_sols, c2_b4_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif', ICP=5, NDIM=n_dim,
                               NPAR=n_params, RL1=25.0, NMX=4000, DSMAX=0.05, name=f'k_pe:tmp',
                               UZR={5: [16.0, 21.0]})

starting_point = 'UZ1'
starting_cont = c2_b4_cont

# continuation of pd-related parameters
#######################################

# continuations of k_gp for k_pe = 16.0
c2_b5_sols, c2_b5_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif', ICP=22, NDIM=n_dim,
                               NPAR=n_params, RL0=0.0, RL1=20.0, NMX=12000, DSMAX=0.05, name=f'k_gp:1',
                               bidirectional=True, UZR={22: [5.0]})
a.run(starting_point='HB1', origin=c2_b5_cont, c='qif2b', ICP=[22, 11], NDIM=n_dim, NPAR=n_params, RL0=10.0, RL1=20.0,
      NMX=2000, DSMAX=0.2, name=f'k_gp:1:lc1', STOP=['BP1'], NPR=10)

# continuations of k_gp for k_pe = 21.0
c2_b6_sols, c2_b6_cont = a.run(starting_point='UZ2', origin=c2_b4_cont, c='qif', ICP=22, NDIM=n_dim,
                               NPAR=n_params, RL0=0.0, RL1=20.0, NMX=12000, DSMAX=0.05, name=f'k_gp:2')
a.run(starting_point='HB1', origin=c2_b6_cont, c='qif2b', ICP=[22, 11], NDIM=n_dim, NPAR=n_params, RL0=15.0, RL1=20.0,
      NMX=2000, DSMAX=0.2, name=f'k_gp:2:lc1', STOP=['BP1', 'LP3'], NPR=10)

# continuations of k_pe for k_gp = 5.0
c2_b7_sols, c2_b7_cont = a.run(starting_point='UZ1', origin=c2_b5_cont, c='qif', ICP=5, NDIM=n_dim,
                               NPAR=n_params, RL0=0.0, RL1=20.0, NMX=16000, DSMAX=0.05, name=f'k_pe:1',
                               DS='-')
a.run(starting_point='HB1', origin=c2_b7_cont, c='qif2b', ICP=[5, 11], NDIM=n_dim, NPAR=n_params, RL0=0.0, RL1=10.0,
      NMX=2000, DSMAX=0.2, name=f'k_pe:1:lc1', STOP=['BP1'], NPR=10)
a.run(starting_point='HB2', origin=c2_b7_cont, c='qif2b', ICP=[5, 11], NDIM=n_dim, NPAR=n_params, RL0=0.0, RL1=10.0,
      NMX=2000, DSMAX=0.2, name=f'k_pe:1:lc2', STOP=['BP1'], NPR=10)

# 2D continuation of k_gp and k_stn
c2_b5_2d1_sols, c2_b5_2d1_cont = a.run(starting_point='HB1', origin=c2_b5_cont, c='qif2', ICP=[5, 22], NDIM=n_dim,
                                       NPAR=n_params, RL0=0.0, RL1=25.0, NMX=20000, DSMAX=0.1,
                                       name=f'k_gp/k_pe:hb1', bidirectional=True)
c2_b7_2d1_sols, c2_b7_2d1_cont = a.run(starting_point='LP2', origin=c2_b7_cont, c='qif2', ICP=[5, 22], NDIM=n_dim,
                                       NPAR=n_params, RL0=0.0, RL1=25.0, NMX=20000, DSMAX=0.1,
                                       name=f'k_gp/k_pe:lp1', bidirectional=True)
c2_b7_2d2_sols, c2_b7_2d2_cont = a.run(starting_point='HB2', origin=c2_b7_cont, c='qif2', ICP=[5, 22], NDIM=n_dim,
                                       NPAR=n_params, RL0=0.0, RL1=25.0, NMX=20000, DSMAX=0.1,
                                       name=f'k_gp/k_pe:hb2', bidirectional=True)

# save results
a.to_file(fname, **kwargs)

# plotting
# ax = a.plot_continuation('PAR(22)', 'PAR(25)', cont='k_gp/k_stn:hb1', line_style_unstable='solid', ignore=['UZ'])
# a.plot_continuation('PAR(22)', 'PAR(25)', cont='k_gp/k_stn:hb2', ax=ax, line_style_unstable='solid',
#                     line_color_stable='#148F77', line_color_unstable='#148F77', ignore=['UZ'])
# # a.plot_continuation('PAR(22)', 'PAR(25)', cont='k_gp/k_stn:hb3', ax=ax, line_style_unstable='solid',
# #                     line_color_stable='#5D6D7E', line_color_unstable='#5D6D7E', ignore=['UZ'])
# a.plot_continuation('PAR(22)', 'PAR(25)', cont='k_gp/k_stn:lp1', ax=ax, line_style_unstable='solid',
#                     line_color_stable='#ee2b2b', line_color_unstable='#ee2b2b', ignore=['UZ'])
# ax.set_xlim([0.0, 25.0])
# ax.set_ylim([0.0, 5.5])
# show()
