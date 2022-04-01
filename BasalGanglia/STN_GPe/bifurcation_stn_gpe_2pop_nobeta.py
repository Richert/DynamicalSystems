from pyrates.utility.pyauto import PyAuto
import sys
from matplotlib.pyplot import show

"""Bifurcation analysis of STN-GPe model with two GPe populations (arkypallidal and prototypical) and 
gamma-distributed axonal delays and bi-exponential synapses."""

path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"

# config
n_dim = 31
n_params = 27
a = PyAuto("auto_files", auto_dir=auto_dir)
kwargs = dict()
model = 'stn_gpe_2pop_nobeta'
fname = f'../results/{model}_conts.pkl'

#########################
# initial continuations #
#########################

# continuation in time
t_sols, t_cont = a.run(e=model, c='ivp', ICP=14, NMX=100000, name='t', UZR={14: 1000.0}, STOP={'UZ1'},
                       NDIM=n_dim, NPAR=n_params)

# set eta_e
s1_sols, s1_cont = a.run(starting_point='UZ1', origin=t_cont, c='qif', ICP=1, NDIM=n_dim,
                         NPAR=n_params, RL0=-11.0, RL1=2.0, NMX=16000, DSMAX=0.05, name=f'eta_e:tmp',
                         UZR={1: [0.9]}, STOP=['UZ1'])

# set eta_p
s2_sols, s2_cont = a.run(starting_point='UZ1', origin=s1_cont, c='qif', ICP=2, NDIM=n_dim,
                         NPAR=n_params, RL0=-11.0, RL1=5.0, NMX=16000, DSMAX=0.05, name=f'eta_p:tmp',
                         UZR={2: [0.35]}, STOP=['UZ1'])

starting_point = 'UZ1'
starting_cont = s2_cont

# continuation of pd-related parameters
#######################################

# continuation of k_gp
c1_sols, c1_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif', ICP=15, NDIM=n_dim,
                         NPAR=n_params, RL0=0.0, RL1=25.0, NMX=12000, DSMAX=0.02, name=f'k_gp:1',
                         bidirectional=True, UZR={15: 15.0})
a.run(starting_point='HB1', origin=c1_cont, c='qif2b', ICP=[15, 11], NDIM=n_dim, NPAR=n_params, RL0=0.0, RL1=10.0,
      NMX=2000, DSMAX=0.05, name=f'k_gp:1:lc1', STOP=['BP1'], NPR=10)

# 2D continuation of k_gp and k_pe
c1_2d1_sols, c1_2d1_cont = a.run(starting_point='HB1', origin=c1_cont, c='qif2', ICP=[4, 15], NDIM=n_dim,
                                 NPAR=n_params, RL0=0.0, RL1=25.0, NMX=10000, DSMAX=0.05,
                                 name=f'k_gp/k_pe:hb1', bidirectional=True, NPR=10)
c1_2d2_sols, c1_2d2_cont = a.run(starting_point='HB3', origin=c1_cont, c='qif2', ICP=[4, 15], NDIM=n_dim,
                                 NPAR=n_params, RL0=0.0, RL1=25.0, NMX=10000, DSMAX=0.05,
                                 name=f'k_gp/k_pe:hb2', bidirectional=True, NPR=10)

# save results
a.to_file(fname, **kwargs)

# plotting
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=2)

ax = axes[0]
ax = a.plot_continuation('PAR(15)', 'PAR(4)', cont='k_gp/k_pe:hb1', line_style_unstable='solid', ignore=['UZ'],
                         ax=ax)
ax = a.plot_continuation('PAR(15)', 'PAR(4)', cont='k_gp/k_pe:hb2', line_style_unstable='solid', ignore=['UZ'],
                         ax=ax, line_color_stable='#148F77', line_color_unstable='#148F77')
ax.set_xlabel(r"$k_{gp}$")
ax.set_ylabel(r"$k_{pe}$")
ax.set_xlim([0.0, 25.0])
ax.set_ylim([0.0, 25.0])

ax = axes[1]
ax = a.plot_continuation('PAR(15)', 'U(3)', cont='k_gp:1', line_style_unstable='solid', ignore=['UZ'],
                         ax=ax)
ax = a.plot_continuation('PAR(15)', 'U(3)', cont='k_gp:1:lc1', line_style_unstable='solid', ignore=['UZ'],
                         ax=ax, line_color_stable='#148F77', line_color_unstable='#148F77')
ax.set_xlabel(r"$k_{gp}$")
ax.set_ylabel(r"$r$")
ax.set_xlim([0.0, 8.0])
ax.set_ylim([0.0, 0.12])

show()
