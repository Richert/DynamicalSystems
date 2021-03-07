from pyrates.utility.pyauto import PyAuto
import sys
import matplotlib.pyplot as plt

"""
Bifurcation analysis of GPe model with two populations (arkypallidal and prototypical) and 
gamma-dstributed axonal delays and bi-exponential synapses. Creates the bifurcation diagrams of Fig. 1 and 2 of
(citation).

To run this code, you need Python >= 3.6 with PyRates (https://github.com/pyrates-neuroscience/PyRates) and 
auto-07p (https://github.com/auto-07p/auto-07p) installed.
"""

path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"

# config
n_dim = 20
n_params = 21
a = PyAuto("auto_files", auto_dir=auto_dir)

# choice of conditions to run bifurcation analysis for
c1 = False  # GPe-p behavior
c2 = False  # GPe-a behavior
c3 = True  # coupled GPe-p and GPe-a

################################
# initial continuation in time #
################################

t_sols, t_cont = a.run(e='gpe_2pop', c='ivp', ICP=14, DS=5e-3, DSMIN=1e-4, DSMAX=1.0, NMX=1000000, name='t',
                       UZR={14: 10000.0}, STOP={'UZ1'}, NDIM=n_dim, NPAR=n_params)

#######################################
# c1: investigation of GPe-p behavior #
#######################################

if c1:

    starting_point = 'UZ1'
    starting_cont = t_cont

    # continuation of eta_p
    #######################

    # step 1: eta_p for k_pp = 0.0
    c1_b1_sols, c1_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params, name='c1:eta_p:1',
                                   NDIM=n_dim, RL0=-60, RL1=60.0, origin=starting_cont, NMX=8000, DSMAX=0.05,
                                   STOP={}, bidirectional=True, UZR={2: 10.0})

    # step 2: k_pp
    c1_b2_sols, c1_b2_cont = a.run(starting_point='UZ1', c='qif', ICP=6, NPAR=n_params, name='c1:k_pp:1',
                                   NDIM=n_dim, RL1=11.0, origin=c1_b1_cont, NMX=8000, DSMAX=0.005,
                                   UZR={6: [1.0, 4.0]}, STOP=[])

    # step 3: eta_p for k_pp = 1.0
    c1_b3_sols, c1_b3_cont = a.run(starting_point='UZ1', c='qif', ICP=2, NPAR=n_params, name='c1:eta_p:2',
                                   NDIM=n_dim, RL0=-50, RL1=50.0, origin=c1_b2_cont, NMX=8000, DSMAX=0.05,
                                   STOP={}, bidirectional=True)

    # step 4: eta_p for k_pp = 2.0
    c1_b4_sols, c1_b4_cont = a.run(starting_point='UZ2', c='qif', ICP=2, NPAR=n_params, name='c1:eta_p:3',
                                   NDIM=n_dim, RL0=-50, RL1=50.0, origin=c1_b2_cont, NMX=8000, DSMAX=0.05,
                                   STOP={}, bidirectional=True)

    # step 4: 2D continuation of Hopf curve
    c1_b4_2d1_sols, c1_b4_2d1_cont = a.run(starting_point='HB1', origin=c1_b4_cont, c='qif2', ICP=[6, 2], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=8000, DSMAX=0.05,
                                           name='c1:eta_p/k_pp:hb1', bidirectional=True)

    # save results
    fname = '../results/gpe_2pop_c1.pkl'
    a.to_file(fname)

    # plotting
    fig, axes = plt.subplots(nrows=2, ncols=2)

    ax1 = axes[0, 0]
    ax1 = a.plot_continuation('PAR(2)', 'PAR(6)', cont='c1:eta_p/k_pp:hb1', ax=ax1)
    ax1.set_xlim([0, 60])
    ax1.set_ylim([0, 6])

    ax2 = axes[0, 1]
    ax2 = a.plot_continuation('PAR(2)', 'U(2)', cont='c1:eta_p:1', ax=ax2)
    ax2.set_title(r'$k_{pp} = 0$')

    ax3 = axes[1, 0]
    ax3 = a.plot_continuation('PAR(2)', 'U(2)', cont='c1:eta_p:2', ax=ax3)
    ax3.set_title(r'$k_{pp} = 1$')

    ax4 = axes[1, 1]
    ax4 = a.plot_continuation('PAR(2)', 'U(2)', cont='c1:eta_p:3', ax=ax4)
    ax4.set_title(r'$k_{pp} = 4$')

    plt.tight_layout()
    plt.show()

#######################################
# c2: investigation of GPe-a behavior #
#######################################

if c2:

    starting_point = 'UZ1'
    starting_cont = t_cont

    # 1D fixed point continuations
    ###############################

    # step 1: eta_a for k_aa = 0.0
    c2_b1_sols, c2_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=3, NPAR=n_params, name='c2:eta_a:1',
                                   NDIM=n_dim, RL0=-50, RL1=50.0, origin=starting_cont, NMX=8000, DSMAX=0.05,
                                   STOP={}, bidirectional=True)

    # step 2: k_pa
    c2_b2_sols, c2_b2_cont = a.run(starting_point=starting_point, c='qif', ICP=9, NPAR=n_params, name='c2:k_aa:1',
                                   NDIM=n_dim, RL1=11.0, origin=starting_cont, NMX=8000, DSMAX=0.005,
                                   UZR={9: [0.1, 1.0]})

    # step 3: eta_a for k_aa = 0.5
    c2_b3_sols, c2_b3_cont = a.run(starting_point='UZ1', c='qif', ICP=3, NPAR=n_params, name='c2:eta_a:2',
                                   NDIM=n_dim, RL0=-50, RL1=50.0, origin=c2_b2_cont, NMX=8000, DSMAX=0.05,
                                   STOP={}, bidirectional=True)

    # step 4: eta_a for k_aa = 1.0
    c2_b4_sols, c2_b4_cont = a.run(starting_point='UZ2', c='qif', ICP=3, NPAR=n_params, name='c2:eta_a:3',
                                   NDIM=n_dim, RL0=-50, RL1=50.0, origin=c2_b2_cont, NMX=8000, DSMAX=0.05,
                                   STOP={}, bidirectional=True)

    # 2D continuation of codim 1 bifurcations
    #########################################

    # step 4: 2D continuation of Hopf curve
    c2_b4_2d1_sols, c2_b4_2d1_cont = a.run(starting_point='HB1', origin=c2_b4_cont, c='qif2', ICP=[9, 3], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=4.0, NMX=8000, DSMAX=0.05,
                                           name='c2:eta_a/k_aa:hb1', bidirectional=True)

    # save results
    fname = '../results/gpe_2pop_c2.pkl'
    a.to_file(fname)

    # plotting
    fig2, axes2 = plt.subplots(nrows=2, ncols=2)

    ax1 = axes2[0, 0]
    ax1 = a.plot_continuation('PAR(3)', 'PAR(9)', cont='c2:eta_a/k_aa:hb1', ax=ax1)
    ax1.set_xlim([0, 50])
    ax1.set_ylim([0, 3])

    # ax2 = a.plot_continuation('PAR(3)', 'U(4)', cont='c2:eta_a:1', ax=ax2, line_color_stable='#148F77',
    #                           line_color_unstable='#148F77')
    ax2 = axes2[0, 1]
    ax2 = a.plot_continuation('PAR(3)', 'U(4)', cont='c2:eta_a:1', ax=ax2)
    ax2.set_title(r'$k_{aa} = 0$')

    ax3 = axes2[1, 0]
    ax3 = a.plot_continuation('PAR(3)', 'U(4)', cont='c2:eta_a:2', ax=ax3)
    ax3.set_title(r'$k_{aa} = 1$')

    ax4 = axes2[1, 1]
    ax4 = a.plot_continuation('PAR(3)', 'U(4)', cont='c2:eta_a:3', ax=ax4)
    ax4.set_title(r'$k_{aa} = 2$')

    plt.tight_layout()
    plt.show()

#################################################################
# c3: investigation of GPe behavior for coupled GPe-p and GPe-a #
#################################################################

if c3:

    starting_point = 'UZ1'
    starting_cont = t_cont

    # set healthy state
    ###################

    # step 1: k_pp
    c3_b1_sols, c3_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=6, NPAR=n_params, name='c3:k_pp:1',
                                   NDIM=n_dim, origin=starting_cont, NMX=8000, DSMAX=0.05, UZR={6: [1.5]},
                                   STOP=['UZ1'])

    # step 2: k_aa
    c3_b2_sols, c3_b2_cont = a.run(starting_point='UZ1', c='qif', ICP=9, NPAR=n_params, name='c3:k_aa:1',
                                   NDIM=n_dim, origin=c3_b1_cont, NMX=8000, DSMAX=0.05, UZR={9: 0.1}, STOP=['UZ1'])

    # step 3: eta_p
    c3_b3_sols, c3_b3_cont = a.run(starting_point='UZ1', c='qif', ICP=2, NPAR=n_params, name='c3:eta_p:1',
                                   NDIM=n_dim, RL0=-50, RL1=50.0, origin=c3_b2_cont, NMX=8000, DSMAX=0.05,
                                   STOP={}, UZR={2: 12.0}, bidirectional=True)

    # step 4: eta_a
    c3_b4_sols, c3_b4_cont = a.run(starting_point='UZ1', c='qif', ICP=3, NPAR=n_params, name='c3:eta_a:1',
                                   NDIM=n_dim, RL0=-50, RL1=50.0, origin=c3_b3_cont, NMX=8000, DSMAX=0.05,
                                   STOP={}, bidirectional=True, UZR={3: 27.0})

    # step 5: k_ap
    c3_b5_sols, c3_b5_cont = a.run(starting_point='UZ1', c='qif', ICP=7, NPAR=n_params, name='c3:k_ap:1',
                                   NDIM=n_dim, origin=c3_b4_cont, NMX=8000, DSMAX=0.05, UZR={7: 2.0}, RL1=2.1)

    # step 6: k_pa
    c3_b6_sols, c3_b6_cont = a.run(starting_point='UZ1', c='qif', ICP=8, NPAR=n_params, name='c3:k_pa:1',
                                   NDIM=n_dim, origin=c3_b5_cont, NMX=8000, DSMAX=0.005, UZR={8: [0.5]}, STOP=['UZ1'])

    # Investigation of oscillations
    ###############################

    # step 1: k_pp
    c3_b7_sols, c3_b7_cont = a.run(starting_point='UZ1', c='qif', ICP=6, NPAR=n_params, name='c3:k_pp:2',
                                   NDIM=n_dim, origin=c3_b6_cont, NMX=8000, DSMAX=0.005, UZR={6: [5.0]}, RL1=6.0)

    # step 2: eta_p
    c3_b8_sols, c3_b8_cont = a.run(starting_point='UZ1', c='qif', ICP=2, NPAR=n_params, name='c3:eta_p:2', RL0=-50.0,
                                   RL1=50.0, NDIM=n_dim, origin=c3_b7_cont, NMX=8000, DSMAX=0.005, bidirectional=True)

    # step 3: 2D continuation of Hopf curve in k_pp and eta_p
    c3_b8_2d1_sols, c3_b8_2d1_cont = a.run(starting_point='HB1', origin=c3_b8_cont, c='qif2', ICP=[6, 2], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=8000, DSMAX=0.01,
                                           name='c3:k_pp/eta_p:hb1', bidirectional=True)

    # step 4: 2D continuation of Hopf curve in k_pa and eta_p
    c3_b8_2d2_sols, c3_b8_2d2_cont = a.run(starting_point='HB1', origin=c3_b8_cont, c='qif2', ICP=[8, 2], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=8000, DSMAX=0.01,
                                           name='c3:k_pa/eta_p:hb1', bidirectional=True)

    # Investigation of bi-stability
    ###############################

    # step 1: find fold
    _, c_tmp = a.run(starting_point='ZH1', c='qif', ICP=2, NPAR=n_params, name='c3:eta_p:3', RL0=-100.0,
                     RL1=100.0, NDIM=n_dim, origin=c3_b8_2d2_cont, NMX=8000, DSMAX=0.005, bidirectional=True)

    # step 2: continue fold curve in k_pa and eta_p
    c3_b8_2d3_sols, c3_b8_2d3_cont = a.run(starting_point='LP1', origin=c_tmp, c='qif2', ICP=[8, 2], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=8000, DSMAX=0.01,
                                           name='c3:k_pa/eta_p:lp1', bidirectional=True)

    # plotting
    fig, axes = plt.subplots(ncols=2)

    ax1 = axes[0]
    ax1 = a.plot_continuation('PAR(2)', 'PAR(6)', cont='c3:k_pp/eta_p:hb1', ax=ax1, line_style_unstable='solid',
                              line_color_stable='#148F77', line_color_unstable='#148F77')
    ax1.set_xlabel(r'$\eta_p$')
    ax1.set_ylabel(r'$k_{pp}$')

    ax2 = axes[1]
    ax2 = a.plot_continuation('PAR(2)', 'PAR(8)', cont='c3:k_pa/eta_p:hb1', ax=ax2, line_style_unstable='solid',
                              line_color_stable='#148F77', line_color_unstable='#148F77')
    ax2 = a.plot_continuation('PAR(2)', 'PAR(8)', cont='c3:k_pa/eta_p:lp1', ax=ax2, line_style_unstable='solid')
    ax2.set_xlabel(r'$\eta_p$')
    ax2.set_ylabel(r'$k_{pa}$')

    plt.tight_layout()
    plt.show()

    # save results
    fname = '../results/gpe_2pop_c3.pkl'
    a.to_file(fname)
