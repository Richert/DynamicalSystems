import sys
import matplotlib.pyplot as plt
from pyrates.utility.pyauto import PyAuto

"""
Bifurcation analysis of GPe model with two populations (arkypallidal and prototypical) and 
gamma-dstributed axonal delays and bi-exponential synapses. Creates data for Fig. 3 of Gast et al. (2021) JNS.

To run this code, you need Python >= 3.6 with PyRates (https://github.com/pyrates-neuroscience/PyRates) and 
auto-07p (https://github.com/auto-07p/auto-07p) installed.
"""

# preparations
##############

path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"

# config
n_dim = 20
n_params = 21
a = PyAuto("auto_files", auto_dir=auto_dir)

# initial continuation in time (to converge to fixed point)
t_sols, t_cont = a.run(e='gpe_2pop', c='ivp', ICP=14, DS=5e-3, DSMIN=1e-4, DSMAX=1.0, NMX=1000000, name='t',
                       UZR={14: 10000.0}, STOP={'UZ1'}, NDIM=n_dim, NPAR=n_params)

########################
# bifurcation analysis #
########################

# set default, healthy state of GPe
###################################

# step 1: k_pp
c1_sols, c1_cont = a.run(starting_point='UZ1', c='qif', ICP=6, NPAR=n_params, NDIM=n_dim, name='k_pp:1',
                         origin=t_cont, NMX=8000, DSMAX=0.05, UZR={6: [6.0]}, STOP=['UZ1'])

# step 2: k_aa
c2_sols, c2_cont = a.run(starting_point='UZ1', c='qif', ICP=9, NPAR=n_params, NDIM=n_dim, name='k_aa:1',
                         origin=c1_cont, NMX=8000, DSMAX=0.05, UZR={9: 0.1}, STOP=['UZ1'])

# step 4: eta_a
c3_sols, c3_cont = a.run(starting_point='UZ1', c='qif', ICP=3, NPAR=n_params, NDIM=n_dim, name='eta_a:1',
                         origin=c2_cont, NMX=8000, DSMAX=0.05, STOP=['UZ1'], UZR={3: 26.0})

# step 5: k_ap
c4_sols, c4_cont = a.run(starting_point='UZ1', c='qif', ICP=7, NPAR=n_params, NDIM=n_dim, name='k_ap:1',
                         origin=c3_cont, NMX=8000, DSMAX=0.05, UZR={7: 2.0}, STOP=['UZ1'])

# step 6: k_pa
c5_sols, c5_cont = a.run(starting_point='UZ1', c='qif', ICP=8, NPAR=n_params, NDIM=n_dim, name='k_pa:1',
                         origin=c4_cont, NMX=8000, DSMAX=0.005, UZR={8: [0.5]}, STOP=['UZ2'])

# Investigation of increased gabaergic decay time constant
##########################################################

# step 1: eta_p
c6_sols, c6_cont = a.run(starting_point='UZ1', c='qif', ICP=2, NPAR=n_params, NDIM=n_dim, name='eta_p:1',
                         origin=c5_cont, NMX=8000, DSMAX=0.05, UZR={2: 50.0})

# step 2: tau_gabaa_d
c7_sols, c7_cont = a.run(starting_point='UZ1', c='qif', ICP=21, NPAR=n_params, NDIM=n_dim, name='tau_gabaa_d:1',
                         origin=c6_cont, NMX=8000, DSMAX=0.005, RL0=0.0, RL1=20.0, bidirectional=True)

# step 2: 2D continuation of Hopf curve in tau_gabaa_d and k_pp
c7_2d1_sols, c7_2d1_cont = a.run(starting_point='HB1', origin=c7_cont, c='qif2', ICP=[20, 21], NDIM=n_dim, NPAR=n_params,
                                 RL0=0.0, RL1=2.0, NMX=8000, DSMAX=0.01, name='tau_gabaa_r/tau_gabaa_d:hb1',
                                 bidirectional=True)

a.run(starting_point='HB1', c='qif2b', ICP=[21, 11], NPAR=n_params, NDIM=n_dim, name='tau_gabaa_d:1:lc',
      RL0=0.5, RL1=20.0,  origin=c7_cont, NMX=2000, DSMAX=0.01)

# save results
fname = '../results/gpe_2pop_syns_conts.pkl'
a.to_file(fname)

# plotting
fig, axes = plt.subplots(ncols=2)

# ax = axes[0]
# ax = a.plot_continuation('PAR(21)', 'U(2)', cont='tau_gabaa_d:1', ax=ax, line_style_unstable='solid',
#                           line_color_stable='#148F77', line_color_unstable='#148F77')
# ax = a.plot_continuation('PAR(21)', 'U(2)', cont='tau_gabaa_d:1:lc', ax=ax, line_style_unstable='solid',
#                           line_color_stable='#148F77', line_color_unstable='#148F77')
# ax2 = ax.twinx()
# ax2 = a.plot_continuation('PAR(21)', 'PAR(11)', cont='tau_gabaa_d:1:lc', ax=ax2, line_style_unstable='solid')
# ax.set_xlabel(r'$\tau_{\mathrm{gabaa}}$')
# ax.set_ylabel(r'$r$')
# ax2.set_ylabel(r'$1/f in ms$')

ax = axes[1]
ax = a.plot_continuation('PAR(21)', 'PAR(20)', cont='tau_gabaa_r/tau_gabaa_d:hb1', ax=ax, line_style_unstable='solid',
                          line_color_stable='#148F77', line_color_unstable='#148F77')
ax.set_xlabel(r'$\tau_{d}$')
ax.set_ylabel(r'$\tau_{r}$')

plt.show()
