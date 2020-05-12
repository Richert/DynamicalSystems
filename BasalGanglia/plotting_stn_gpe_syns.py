import matplotlib.pyplot as plt
import matplotlib as mpl
from pyauto import PyAuto
import matplotlib.gridspec as gs

# plotting parameters
# linewidth = 2.0
# fontsize1 = 12
# fontsize2 = 14
# markersize1 = 80
# markersize2 = 80
linewidth = 1.2
fontsize1 = 10
fontsize2 = 12
markersize1 = 60
markersize2 = 60
dpi = 200

plt.style.reload_library()
plt.style.use('seaborn-whitegrid')
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
#mpl.rc('text', usetex=True)
mpl.rcParams["font.sans-serif"] = ["Roboto"]
mpl.rcParams["font.size"] = fontsize1
mpl.rcParams["font.weight"] = "bold"
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['axes.titlesize'] = fontsize2
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelsize'] = 'large'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['ytick.alignment'] = 'center'
mpl.rcParams['legend.fontsize'] = fontsize1


############################################
# file loading and condition specification #
############################################

fname = 'results/stn_gpe_syns.pkl'
a = PyAuto.from_file(fname)

c1 = [  # strong GPe-p projections
      True,  # STN -> GPe-p > STN -> GPe-a
      True,   # STN -> GPe-p == STN -> GPe-a
]
c2 = False  # strong bidirectional coupling between GPe-p and GPe-a
c3 = False  # strong GPe-p to GPe-a projection

##########################################################################
# c1: investigation of GPe behavior for strong GPe-p to GPe-a connection #
##########################################################################

if any(c1):

    if c1[0]:

        # continuation of eta_e
        #######################

        fig1 = plt.figure(tight_layout=True, figsize=(6.0, 9.0), dpi=dpi)
        grid1 = gs.GridSpec(2, 2)

        # codim 1
        ax = fig1.add_subplot(grid1[0, :])
        ax = a.plot_continuation('PAR(1)', 'U(3)', cont='c1:eta_e', ax=ax, line_color_stable='#76448A',
                                 line_color_unstable='#5D6D7E', default_size=markersize1)
        ax = a.plot_continuation('PAR(1)', 'U(3)', cont='c1:eta_e_lc', ax=ax, ignore=['BP'], line_color_stable='#148F77',
                                 default_size=markersize1, custom_bf_styles={'LP': {'marker': 'p'}})
        ax.set_xlabel(r'$\eta_e$')
        ax.set_ylabel('Firing rate (GPe-p)')

        # codim 2
        ax = fig1.add_subplot(grid1[1, 0])
        ax = a.plot_continuation('PAR(1)', 'PAR(2)', cont='c1:eta_e/eta_p', ax=ax, line_color_stable='#8299b0',
                                 line_color_unstable='#8299b0', default_size=markersize1, line_style_unstable='solid')
        ax = a.plot_continuation('PAR(1)', 'PAR(2)', cont='c1:eta_e/eta_p/gh1', ax=ax, line_color_stable='#3689c9',
                                 line_color_unstable='#3689c9', default_size=markersize1, line_style_unstable='solid')
        ax = a.plot_continuation('PAR(1)', 'PAR(2)', cont='c1:eta_e/eta_p/gh2', ax=ax, line_color_stable='#3689c9',
                                 line_color_unstable='#3689c9', default_size=markersize1, line_style_unstable='solid')
        ax.set_xlabel(r'$\eta_e$')
        ax.set_ylabel(r'$\eta_p$')

        ax = fig1.add_subplot(grid1[1, 1])
        ax = a.plot_continuation('PAR(1)', 'PAR(3)', cont='c1:eta_e/eta_a', ax=ax, line_color_stable='#8299b0',
                                 line_color_unstable='#8299b0', default_size=markersize1, line_style_unstable='solid')
        ax = a.plot_continuation('PAR(1)', 'PAR(3)', cont='c1:eta_e/eta_a/zh1', ax=ax, line_color_stable='#3689c9',
                                 line_color_unstable='#3689c9', default_size=markersize1, line_style_unstable='solid')
        #ax = a.plot_continuation('PAR(1)', 'PAR(3)', cont='c1:eta_e/eta_a/bt1', ax=ax, line_color_stable='#b8b632',
        #                         line_color_unstable='#b8b632', default_size=markersize1, line_style_unstable='solid')
        ax.set_xlabel(r'$\eta_e$')
        ax.set_ylabel(r'$\eta_a$')

        plt.tight_layout()

        # continuation of eta_a
        #######################

        fig2 = plt.figure(tight_layout=True, figsize=(6.0, 9.0), dpi=dpi)
        grid2 = gs.GridSpec(2, 2)

        # codim 1
        ax = fig2.add_subplot(grid2[0, :])
        ax = a.plot_continuation('PAR(3)', 'U(3)', cont='c1:eta_a', ax=ax, line_color_stable='#76448A',
                                 line_color_unstable='#5D6D7E', default_size=markersize1)
        ax.set_xlabel(r'$\eta_a$')
        ax.set_ylabel('Firing rate (GPe-p)')

        # codim 2
        ax = fig2.add_subplot(grid2[1, 0])
        ax = a.plot_continuation('PAR(3)', 'PAR(1)', cont='c1:eta_a/eta_e', ax=ax, line_color_stable='#8299b0',
                                 line_color_unstable='#8299b0', default_size=markersize1, line_style_unstable='solid')
        ax = a.plot_continuation('PAR(3)', 'PAR(1)', cont='c1:eta_a/eta_e/bt1', ax=ax, line_color_stable='#3689c9',
                                 line_color_unstable='#3689c9', default_size=markersize1, line_style_unstable='solid')
        ax = a.plot_continuation('PAR(3)', 'PAR(1)', cont='c1:eta_a/eta_e/bt2', ax=ax, line_color_stable='#b8b632',
                                 line_color_unstable='#b8b632', default_size=markersize1, line_style_unstable='solid')
        ax.set_xlabel(r'$\eta_a$')
        ax.set_ylabel(r'$\eta_e$')

        ax = fig2.add_subplot(grid2[1, 1])
        ax = a.plot_continuation('PAR(3)', 'PAR(2)', cont='c1:eta_a/eta_p', ax=ax, line_color_stable='#8299b0',
                                 line_color_unstable='#8299b0', default_size=markersize1, line_style_unstable='solid')
        ax = a.plot_continuation('PAR(3)', 'PAR(2)', cont='c1:eta_a/eta_p/zh1', ax=ax, line_color_stable='#3689c9',
                                 line_color_unstable='#3689c9', default_size=markersize1, line_style_unstable='solid')
        ax = a.plot_continuation('PAR(3)', 'PAR(2)', cont='c1:eta_a/eta_p/zh2', ax=ax, line_color_stable='#b8b632',
                                 line_color_unstable='#b8b632', default_size=markersize1, line_style_unstable='solid')
        ax.set_xlabel(r'$\eta_a$')
        ax.set_ylabel(r'$\eta_p$')

        plt.tight_layout()

        plt.show()

    if c1[1]:

        # continuation of eta_e
        #######################

        fig1 = plt.figure(tight_layout=True, figsize=(6.0, 9.0), dpi=dpi)
        grid1 = gs.GridSpec(3, 2)

        # codim 1: eta_e
        ax = fig1.add_subplot(grid1[0, :])
        ax = a.plot_continuation('PAR(1)', 'U(3)', cont='c1.2:eta_e', ax=ax, line_color_stable='#76448A',
                                 line_color_unstable='#5D6D7E', default_size=markersize1)
        ax = a.plot_continuation('PAR(1)', 'U(3)', cont='c1.2:eta_e_lc1', ax=ax, ignore=['BP'],
                                 line_color_stable='#148F77',
                                 default_size=markersize1, custom_bf_styles={'LP': {'marker': 'p'}})
        ax = a.plot_continuation('PAR(1)', 'U(3)', cont='c1.2:eta_e_lc2', ax=ax, ignore=['BP'],
                                 line_color_stable='#148F77',
                                 default_size=markersize1, custom_bf_styles={'LP': {'marker': 'p'}})
        ax.set_xlabel(r'$\eta_e$')
        ax.set_ylabel('Firing rate (GPe-p)')

        # codim 2: HB3
        ax = fig1.add_subplot(grid1[1, 0])
        ax = a.plot_continuation('PAR(1)', 'PAR(2)', cont='c1.2:eta_e/eta_p/hb1', ax=ax, line_color_stable='#8299b0',
                                 line_color_unstable='#8299b0', default_size=markersize1, line_style_unstable='solid')
        # ax = a.plot_continuation('PAR(1)', 'PAR(2)', cont='c1:eta_e/eta_p/gh1', ax=ax, line_color_stable='#3689c9',
        #                          line_color_unstable='#3689c9', default_size=markersize1, line_style_unstable='solid')
        # ax = a.plot_continuation('PAR(1)', 'PAR(2)', cont='c1:eta_e/eta_p/gh2', ax=ax, line_color_stable='#3689c9',
        #                          line_color_unstable='#3689c9', default_size=markersize1, line_style_unstable='solid')
        ax.set_xlabel(r'$\eta_e$')
        ax.set_ylabel(r'$\eta_p$')

        ax = fig1.add_subplot(grid1[1, 1])
        ax = a.plot_continuation('PAR(1)', 'PAR(3)', cont='c1.2:eta_e/eta_a/hb1', ax=ax, line_color_stable='#8299b0',
                                 line_color_unstable='#8299b0', default_size=markersize1, line_style_unstable='solid')
        # ax = a.plot_continuation('PAR(1)', 'PAR(3)', cont='c1:eta_e/eta_a/zh1', ax=ax, line_color_stable='#3689c9',
        #                          line_color_unstable='#3689c9', default_size=markersize1, line_style_unstable='solid')
        # ax = a.plot_continuation('PAR(1)', 'PAR(3)', cont='c1:eta_e/eta_a/bt1', ax=ax, line_color_stable='#b8b632',
        #                          line_color_unstable='#b8b632', default_size=markersize1, line_style_unstable='solid')
        ax.set_xlabel(r'$\eta_e$')
        ax.set_ylabel(r'$\eta_a$')

        # codim 2: LP1
        ax = fig1.add_subplot(grid1[2, 0])
        ax = a.plot_continuation('PAR(1)', 'PAR(2)', cont='c1.2:eta_e/eta_p/lp1', ax=ax, line_color_stable='#8299b0',
                                 line_color_unstable='#8299b0', default_size=markersize1, line_style_unstable='solid')
        # ax = a.plot_continuation('PAR(1)', 'PAR(2)', cont='c1:eta_e/eta_p/gh1', ax=ax, line_color_stable='#3689c9',
        #                          line_color_unstable='#3689c9', default_size=markersize1, line_style_unstable='solid')
        # ax = a.plot_continuation('PAR(1)', 'PAR(2)', cont='c1:eta_e/eta_p/gh2', ax=ax, line_color_stable='#3689c9',
        #                          line_color_unstable='#3689c9', default_size=markersize1, line_style_unstable='solid')
        ax.set_xlabel(r'$\eta_e$')
        ax.set_ylabel(r'$\eta_p$')

        ax = fig1.add_subplot(grid1[2, 1])
        ax = a.plot_continuation('PAR(1)', 'PAR(3)', cont='c1.2:eta_e/eta_a/lp1', ax=ax, line_color_stable='#8299b0',
                                 line_color_unstable='#8299b0', default_size=markersize1, line_style_unstable='solid')
        # ax = a.plot_continuation('PAR(1)', 'PAR(3)', cont='c1:eta_e/eta_a/zh1', ax=ax, line_color_stable='#3689c9',
        #                          line_color_unstable='#3689c9', default_size=markersize1, line_style_unstable='solid')
        # ax = a.plot_continuation('PAR(1)', 'PAR(3)', cont='c1:eta_e/eta_a/bt1', ax=ax, line_color_stable='#b8b632',
        #                          line_color_unstable='#b8b632', default_size=markersize1, line_style_unstable='solid')
        ax.set_xlabel(r'$\eta_e$')
        ax.set_ylabel(r'$\eta_a$')

        plt.tight_layout()
        plt.show()

###############################################################################################
# c2: investigation of GPe behavior for strong bidirectional coupling between GPe-p and GPe-a #
###############################################################################################

if c2:

    # continuation of eta_p
    #######################

    fig3 = plt.figure(tight_layout=True, figsize=(6.0, 9.0), dpi=dpi)
    grid3 = gs.GridSpec(3, 2)

    # codim 1
    ax = fig3.add_subplot(grid3[0, :])
    ax = a.plot_continuation('PAR(2)', 'U(2)', cont='c2:eta_p', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\eta_p$')
    ax.set_ylabel('Firing rate (GPe-p)')

    # codim 2
    ax = fig3.add_subplot(grid3[1, 0])
    ax = a.plot_continuation('PAR(2)', 'PAR(3)', cont='c2:eta_p/eta_a', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\eta_p$')
    ax.set_ylabel(r'$\eta_a$')

    ax = fig3.add_subplot(grid3[1, 1])
    ax = a.plot_continuation('PAR(2)', 'PAR(19)', cont='c2:eta_p/k_gp', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\eta_a$')
    ax.set_ylabel(r'$k_{gp}$')

    ax = fig3.add_subplot(grid3[2, 0])
    ax = a.plot_continuation('PAR(2)', 'PAR(20)', cont='c2:eta_p/k_p', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\eta_a$')
    ax.set_ylabel(r'$k_{p}$')

    ax = fig3.add_subplot(grid3[2, 1])
    ax = a.plot_continuation('PAR(2)', 'PAR(21)', cont='c2:eta_p/k_i', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\eta_a$')
    ax.set_ylabel(r'$k_{i}$')

    plt.tight_layout()

    # continuation of delta_a
    #########################

    fig4 = plt.figure(tight_layout=True, figsize=(6.0, 9.0), dpi=dpi)
    grid4 = gs.GridSpec(3, 2)

    # codim 1
    ax = fig4.add_subplot(grid4[0, :])
    ax = a.plot_continuation('PAR(17)', 'U(2)', cont='c2:delta_a', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\Delta_a$')
    ax.set_ylabel('Firing rate (GPe-p)')

    # codim 2
    ax = fig4.add_subplot(grid4[1, 0])
    ax = a.plot_continuation('PAR(17)', 'PAR(16)', cont='c2:delta_a/delta_p', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\Delta_a$')
    ax.set_ylabel(r'$\Delta_p$')

    ax = fig4.add_subplot(grid4[1, 1])
    ax = a.plot_continuation('PAR(17)', 'PAR(2)', cont='c2:delta_a/eta_p', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\Delta_a$')
    ax.set_ylabel(r'$\eta_p$')

    ax = fig4.add_subplot(grid4[2, 0])
    ax = a.plot_continuation('PAR(17)', 'PAR(3)', cont='c2:delta_a/eta_a', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\Delta_a$')
    ax.set_ylabel(r'$\eta_a$')

    ax = fig4.add_subplot(grid4[2, 1])
    ax = a.plot_continuation('PAR(17)', 'PAR(21)', cont='c2:delta_a/k_i', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\Delta_a$')
    ax.set_ylabel(r'$k_i$')

    plt.tight_layout()

    plt.show()

##########################################################################
# c3: investigation of GPe behavior for strong GPe-p to GPe-a connection #
##########################################################################

if c3:

    # continuation of delta_p
    #########################

    fig5 = plt.figure(tight_layout=True, figsize=(6.0, 9.0), dpi=dpi)
    grid5 = gs.GridSpec(3, 2)

    # codim 1
    ax = fig5.add_subplot(grid5[0, :])
    ax = a.plot_continuation('PAR(16)', 'U(2)', cont='c3:delta_p', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax = a.plot_continuation('PAR(16)', 'U(2)', cont='c3:delta_p_lc', ax=ax, ignore=['BP'], line_color_stable='#148F77',
                             default_size=markersize1, custom_bf_styles={'LP': {'marker': 'p'}})
    ax.set_xlabel(r'$\Delta_p$')
    ax.set_ylabel('Firing rate (GPe-p)')

    # codim 2
    ax = fig5.add_subplot(grid5[1, 0])
    ax = a.plot_continuation('PAR(16)', 'PAR(3)', cont='c3:delta_p/eta_a', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\Delta_p$')
    ax.set_ylabel(r'$\eta_a$')

    ax = fig5.add_subplot(grid5[1, 1])
    ax = a.plot_continuation('PAR(16)', 'PAR(2)', cont='c3:delta_p/eta_p', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\Delta_p$')
    ax.set_ylabel(r'$\eta_p$')

    ax = fig5.add_subplot(grid5[2, 0])
    ax = a.plot_continuation('PAR(16)', 'PAR(21)', cont='c3:delta_p/k_i', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\Delta_p$')
    ax.set_ylabel(r'$\frac{k_{inter}}{k_{intra}}$')

    ax = fig5.add_subplot(grid5[2, 1])
    ax = a.plot_continuation('PAR(16)', 'PAR(22)', cont='c3:delta_p/k_pi', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\Delta_p$')
    ax.set_ylabel(r'$\frac{k_{ap}}{k_{pa}}$')
    plt.tight_layout()

    # continuation of eta_p
    #######################

    fig6 = plt.figure(tight_layout=True, figsize=(6.0, 9.0), dpi=dpi)
    grid6 = gs.GridSpec(3, 2)

    # codim 1
    ax = fig6.add_subplot(grid6[0, :])
    ax = a.plot_continuation('PAR(2)', 'U(2)', cont='c3:eta_p', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax = a.plot_continuation('PAR(2)', 'U(2)', cont='c3:eta_p_lc', ax=ax, ignore=['BP'], line_color_stable='#148F77',
                             default_size=markersize1, custom_bf_styles={'LP': {'marker': 'p'}})
    ax.set_xlabel(r'$\eta_p$')
    ax.set_ylabel('Firing rate (GPe-p)')

    # codim 2
    ax = fig6.add_subplot(grid6[1, 0])
    ax = a.plot_continuation('PAR(2)', 'PAR(3)', cont='c3:eta_p/eta_a', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\eta_p$')
    ax.set_ylabel(r'$\eta_a$')

    ax = fig6.add_subplot(grid6[1, 1])
    ax = a.plot_continuation('PAR(2)', 'PAR(19)', cont='c3:eta_p/k_gp', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\eta_p$')
    ax.set_ylabel(r'$k_{all}$')

    ax = fig6.add_subplot(grid6[2, 0])
    ax = a.plot_continuation('PAR(2)', 'PAR(21)', cont='c3:eta_p/k_i', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\eta_p$')
    ax.set_ylabel(r'$\frac{k_{inter}}{k_{intra}}$')

    ax = fig6.add_subplot(grid6[2, 1])
    ax = a.plot_continuation('PAR(2)', 'PAR(22)', cont='c3:eta_p/k_pi', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\eta_p$')
    ax.set_ylabel(r'$\frac{k_{ap}}{k_{pa}}$')
    plt.tight_layout()

    # continuation of delta_p
    #########################

    fig7 = plt.figure(tight_layout=True, figsize=(6.0, 9.0), dpi=dpi)
    grid7 = gs.GridSpec(3, 2)

    # codim 1
    ax = fig7.add_subplot(grid7[0, :])
    ax = a.plot_continuation('PAR(3)', 'U(2)', cont='c3:eta_a', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax = a.plot_continuation('PAR(3)', 'U(2)', cont='c3:eta_a_lc', ax=ax, ignore=['BP'], line_color_stable='#148F77',
                             default_size=markersize1, custom_bf_styles={'LP': {'marker': 'p'}})
    ax.set_xlabel(r'$\eta_a$')
    ax.set_ylabel('Firing rate (GPe-p)')

    # codim 2
    ax = fig7.add_subplot(grid7[1, 0])
    ax = a.plot_continuation('PAR(3)', 'PAR(2)', cont='c3:eta_a/eta_p', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\eta_a$')
    ax.set_ylabel(r'$\eta_p$')

    ax = fig7.add_subplot(grid7[1, 1])
    ax = a.plot_continuation('PAR(3)', 'PAR(19)', cont='c3:eta_a/k_gp', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\eta_a$')
    ax.set_ylabel(r'$k_{all}$')

    ax = fig7.add_subplot(grid7[2, 0])
    ax = a.plot_continuation('PAR(3)', 'PAR(21)', cont='c3:eta_a/k_i', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\eta_a$')
    ax.set_ylabel(r'$\frac{k_{inter}}{k_{intra}}$')

    ax = fig7.add_subplot(grid7[2, 1])
    ax = a.plot_continuation('PAR(3)', 'PAR(22)', cont='c3:eta_a/k_pi', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\eta_a$')
    ax.set_ylabel(r'$\frac{k_{ap}}{k_{pa}}$')
    plt.tight_layout()

    plt.show()
