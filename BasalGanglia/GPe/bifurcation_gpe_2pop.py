import sys
import matplotlib.pyplot as plt
from pyrates.utility.pyauto import PyAuto

"""
Bifurcation analysis of GPe model with two populations (arkypallidal and prototypical) and 
gamma-dstributed axonal delays and bi-exponential synapses. Creates data for Fig. 2 of Gast et al. (2021) JNS.

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
                         origin=t_cont, NMX=8000, DSMAX=0.05, UZR={6: [1.5]}, STOP=['UZ1'])

# step 2: k_aa
c2_sols, c2_cont = a.run(starting_point='UZ1', c='qif', ICP=9, NPAR=n_params, NDIM=n_dim, name='k_aa:1',
                         origin=c1_cont, NMX=8000, DSMAX=0.05, UZR={9: 0.1}, STOP=['UZ1'])

# step 3: eta_p
c3_sols, c3_cont = a.run(starting_point='UZ1', c='qif', ICP=2, NPAR=n_params, NDIM=n_dim, name='eta_p:1',
                         origin=c2_cont, NMX=8000, DSMAX=0.05, STOP=['UZ1'], UZR={2: 12.0})

# step 4: eta_a
c4_sols, c4_cont = a.run(starting_point='UZ1', c='qif', ICP=3, NPAR=n_params, NDIM=n_dim, name='eta_a:1',
                         origin=c3_cont, NMX=8000, DSMAX=0.05, STOP=['UZ1'], bidirectional=True, UZR={3: 26.0})

# step 5: k_ap
c5_sols, c5_cont = a.run(starting_point='UZ1', c='qif', ICP=7, NPAR=n_params, NDIM=n_dim, name='k_ap:1',
                         origin=c4_cont, NMX=8000, DSMAX=0.05, UZR={7: 2.0}, STOP=['UZ1'])

# step 6: k_pa
c6_sols, c6_cont = a.run(starting_point='UZ1', c='qif', ICP=8, NPAR=n_params, NDIM=n_dim, name='k_pa:1',
                         origin=c5_cont, NMX=8000, DSMAX=0.005, UZR={8: [0.1, 5.0]}, STOP=['UZ2'])

# Investigation of healthy state I-O curve
##########################################

# step 1: eta_p
c0_sols, c0_cont = a.run(starting_point='UZ1', c='qif', ICP=2, NPAR=n_params, NDIM=n_dim, name='eta_p:0',
                         origin=c6_cont, NMX=8000, DSMAX=0.05, STOP=['UZ1'], bidirectional=True)

# Investigation of increased GPe-p self-inhibition
##################################################

# step 1: k_pp
c7_sols, c7_cont = a.run(starting_point='UZ1', c='qif', ICP=6, NPAR=n_params, NDIM=n_dim, name='k_pp:2',
                         origin=c6_cont, NMX=8000, DSMAX=0.005, UZR={6: [5.0]}, RL1=6.0)

# step 2: eta_p
c8_sols, c8_cont = a.run(starting_point='UZ1', c='qif', ICP=2, NPAR=n_params, NDIM=n_dim, name='eta_p:2',
                         RL0=-50.0, RL1=50.0,  origin=c7_cont, NMX=8000, DSMAX=0.005, bidirectional=True)
a.run(starting_point='HB1', c='qif2b', ICP=[2, 11], NPAR=n_params, NDIM=n_dim, name='eta_p:2:lc',
      RL1=50.0,  origin=c8_cont, NMX=2000, DSMAX=0.01)

# step 3: 2D continuation of Hopf curve in k_pp and eta_p
c8_2d1_sols, c8_2d1_cont = a.run(starting_point='HB1', origin=c8_cont, c='qif2', ICP=[6, 2], NDIM=n_dim, NPAR=n_params,
                                 RL0=0.0, RL1=10.0, NMX=8000, DSMAX=0.01, name='k_pp/eta_p:hb1', bidirectional=True)

# step 4: continue hopf curve in k_pa and eta_p
c8_2d2_sols, c8_2d2_cont = a.run(starting_point='HB1', origin=c8_cont, c='qif2', ICP=[8, 2], NDIM=n_dim,
                                 NPAR=n_params, RL0=0.0, RL1=10.0, NMX=8000, DSMAX=0.01, name='k_pa/eta_p:hb1',
                                 bidirectional=True)

# Investigation of increased GPe-a inhibition of GPe-p
######################################################

# step 1: eta_p
c9_sols, c9_cont = a.run(starting_point='UZ2', c='qif', ICP=2, NPAR=n_params, NDIM=n_dim, name='eta_p:3',
                         RL0=-50.0, RL1=50.0,  origin=c6_cont, NMX=8000, DSMAX=0.005, bidirectional=True)

# step 2: continue fold curve in k_pa and eta_p
c9_2d1_sols, c9_2d1_cont = a.run(starting_point='LP1', origin=c9_cont, c='qif2', ICP=[8, 2], NDIM=n_dim, NPAR=n_params,
                                 RL0=0.0, RL1=10.0, NMX=8000, DSMAX=0.01, name='k_pa/eta_p:lp1', bidirectional=True)

# step 3: continue hopf curve in k_pa and eta_p
c9_2d2_sols, c9_2d2_cont = a.run(starting_point='HB1', origin=c9_cont, c='qif2', ICP=[8, 2], NDIM=n_dim,
                                 NPAR=n_params, RL0=0.0, RL1=10.0, NMX=16000, DSMAX=0.002, name='k_pa/eta_p:hb1',
                                 bidirectional=True)

# step 4: continue the limit cycles of the hopf bifurcations
c10_sols, c10_cont = a.run(starting_point='HB2', c='qif2b', ICP=[2, 11], NPAR=n_params, NDIM=n_dim, name='eta_p:3:lc',
                           RL1=50.0,  origin=c9_cont, NMX=2000, DSMAX=0.01, STOP=['BP1'])

# # step 5: continue the locus of the fold of limit cycle bifurcation in k_pa and eta_p
# c10_2d1_sols, c10_2d1_cont = a.run(starting_point='LP1', origin=c10_cont, c='qif3', ICP=[8, 2, 11], NDIM=n_dim,
#                                    NPAR=n_params, RL0=0.0, RL1=10.0, NMX=2000, DSMAX=0.1, name='k_pa/eta_p:lc/lp1',
#                                    DS='-')
#
# # step 6: continue the locus of the torus bifurcation in k_pa and eta_p
# c10_2d2_sols, c10_2d2_cont = a.run(starting_point='TR1', origin=c10_cont, c='qif3', ICP=[8, 2, 11], NDIM=n_dim,
#                                    NPAR=n_params, RL0=0.0, RL1=10.0, NMX=2000, DSMAX=0.1, name='k_pa/eta_p:lc/tr1',
#                                    DS='-', STOP='BP1')

# save results
fname = '../results/gpe_2pop_conts.pkl'
a.to_file(fname)

# plotting
fig, axes = plt.subplots(ncols=2, nrows=2)

ax1 = axes[0, 0]
ax1 = a.plot_continuation('PAR(2)', 'PAR(6)', cont='k_pp/eta_p:hb1', ax=ax1, line_style_unstable='solid',
                          line_color_stable='#148F77', line_color_unstable='#148F77')
ax1.set_xlabel(r'$\eta_p$')
ax1.set_ylabel(r'$k_{pp}$')

ax2 = axes[0, 1]
ax2 = a.plot_continuation('PAR(2)', 'PAR(8)', cont='k_pa/eta_p:lp1', ax=ax2, line_style_unstable='solid')
ax2 = a.plot_continuation('PAR(2)', 'PAR(8)', cont='k_pa/eta_p:hb1', ax=ax2, line_style_unstable='solid',
                          line_color_stable='#148F77', line_color_unstable='#148F77')
# ax2 = a.plot_continuation('PAR(2)', 'PAR(8)', cont='k_pa/eta_p:lc/lp1', ax=ax2, line_style_unstable='solid',
#                           line_color_stable='#ee2b2b', line_color_unstable='#ee2b2b')
# ax2 = a.plot_continuation('PAR(2)', 'PAR(8)', cont='k_pa/eta_p:lc/tr1', ax=ax2, line_style_unstable='solid',
#                           line_color_stable='#3689c9', line_color_unstable='#3689c9')
ax2.set_xlabel(r'$\eta_p$')
ax2.set_ylabel(r'$k_{pa}$')

ax3 = axes[1, 0]
ax3 = a.plot_continuation('PAR(2)', 'U(2)', cont='eta_p:2', ax=ax3)
ax3 = a.plot_continuation('PAR(2)', 'U(4)', cont='eta_p:2', ax=ax3, line_color_stable='#148F77',
                          line_color_unstable='#148F77')
ax3 = a.plot_continuation('PAR(2)', 'U(2)', cont='eta_p:2:lc', ax=ax3)
ax3 = a.plot_continuation('PAR(2)', 'U(4)', cont='eta_p:2:lc', ax=ax3, line_color_stable='#148F77',
                          line_color_unstable='#148F77')
ax3.set_xlabel(r'$\eta_p$')
ax3.set_ylabel(r'$r$')

ax4 = axes[1, 1]
ax4 = a.plot_continuation('PAR(2)', 'U(2)', cont='eta_p:3', ax=ax4)
ax4 = a.plot_continuation('PAR(2)', 'U(2)', cont='eta_p:3:lc', ax=ax4, line_color_stable='#148F77',
                          line_color_unstable='#148F77')
ax4 = a.plot_continuation('PAR(2)', 'U(2)', cont='eta_p:3:lc', ax=ax4, line_color_stable='#3689c9',
                          line_color_unstable='#3689c9')
ax4.set_xlabel(r'$\eta_p$')
ax4.set_ylabel(r'$r$')

plt.tight_layout()
plt.show()
