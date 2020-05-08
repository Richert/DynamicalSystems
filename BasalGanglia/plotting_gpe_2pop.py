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

fname = 'results/gpe_2pop.pkl'
a = PyAuto.from_file(fname)

c1 = False  # strong GPe-p projections
c2 = False  # strong bidirectional coupling between GPe-p and GPe-a
c3 = True  # strong GPe-p to GPe-a projection

##########################################################################
# c1: investigation of GPe behavior for strong GPe-p to GPe-a connection #
##########################################################################

if c1:

    # continuation of eta_a
    #######################

    fig1 = plt.figure(tight_layout=True, figsize=(8.0, 6.0), dpi=dpi)
    grid1 = gs.GridSpec(3, 2)

    # codim 1
    ax = fig1.add_subplot(grid1[0, :])
    ax = a.plot_continuation('PAR(3)', 'U(2)', cont='c1:eta_a', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\eta_a$')
    ax.set_ylabel('Firing rate (GPe-p)')

    # codim 2
    ax = fig1.add_subplot(grid1[1, 0])
    ax = a.plot_continuation('PAR(3)', 'PAR(2)', cont='c1:eta_a/eta_p', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\eta_a$')
    ax.set_ylabel(r'$\eta_p$')

    ax = fig1.add_subplot(grid1[1, 1])
    ax = a.plot_continuation('PAR(3)', 'PAR(19)', cont='c1:eta_a/k_gp', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\eta_a$')
    ax.set_ylabel(r'$k_{gp}$')

    ax = fig1.add_subplot(grid1[2, 0])
    ax = a.plot_continuation('PAR(3)', 'PAR(20)', cont='c1:eta_a/k_p', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\eta_a$')
    ax.set_ylabel(r'$k_{p}$')

    ax = fig1.add_subplot(grid1[2, 1])
    ax = a.plot_continuation('PAR(3)', 'PAR(21)', cont='c1:eta_a/k_i', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\eta_a$')
    ax.set_ylabel(r'$k_{i}$')

    plt.tight_layout()

    # continuation of delta_p
    #########################

    fig2 = plt.figure(tight_layout=True, figsize=(8.0, 6.0), dpi=dpi)
    grid2 = gs.GridSpec(3, 2)

    # codim 1
    ax = fig2.add_subplot(grid2[0, :])
    ax = a.plot_continuation('PAR(16)', 'U(2)', cont='c1:delta_p', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax = a.plot_continuation('PAR(16)', 'U(2)', cont='c1:delta_p_lc', ax=ax, ignore=['BP'], line_color_stable='#148F77',
                             default_size=markersize1, custom_bf_styles={'LP': {'marker': 'p'}})
    ax.set_xlabel(r'$\Delta_p$')
    ax.set_ylabel('Firing rate (GPe-p)')

    # codim 2
    ax = fig2.add_subplot(grid2[1, 0])
    ax = a.plot_continuation('PAR(16)', 'PAR(17)', cont='c1:delta_p/delta_a', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\Delta_p$')
    ax.set_ylabel(r'$\Delta_a$')

    ax = fig2.add_subplot(grid2[1, 1])
    ax = a.plot_continuation('PAR(16)', 'PAR(2)', cont='c1:delta_p/eta_p', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\Delta_p$')
    ax.set_ylabel(r'$\eta_p$')

    ax = fig2.add_subplot(grid2[2, 0])
    ax = a.plot_continuation('PAR(16)', 'PAR(3)', cont='c1:delta_p/eta_a', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\Delta_p$')
    ax.set_ylabel(r'$\eta_a$')

    ax = fig2.add_subplot(grid2[2, 1])
    ax = a.plot_continuation('PAR(16)', 'PAR(20)', cont='c1:delta_p/k_p', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\Delta_p$')
    ax.set_ylabel(r'$k_p$')

    plt.tight_layout()

    plt.show()

###############################################################################################
# c2: investigation of GPe behavior for strong bidirectional coupling between GPe-p and GPe-a #
###############################################################################################

if c2:

    # continuation of eta_p
    #######################

    fig3 = plt.figure(tight_layout=True, figsize=(8.0, 6.0), dpi=dpi)
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

    fig4 = plt.figure(tight_layout=True, figsize=(8.0, 6.0), dpi=dpi)
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

    fig5 = plt.figure(tight_layout=True, figsize=(8.0, 6.0), dpi=dpi)
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
    ax = a.plot_continuation('PAR(16)', 'PAR(17)', cont='c3:delta_p/delta_a', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\Delta_p$')
    ax.set_ylabel(r'$\Delta_a$')

    ax = fig5.add_subplot(grid5[1, 1])
    ax = a.plot_continuation('PAR(16)', 'PAR(2)', cont='c3:delta_p/eta_p', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\Delta_p$')
    ax.set_ylabel(r'$\eta_p$')

    ax = fig5.add_subplot(grid5[2, 0])
    ax = a.plot_continuation('PAR(16)', 'PAR(3)', cont='c3:delta_p/eta_a', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\Delta_p$')
    ax.set_ylabel(r'$\eta_a$')

    ax = fig5.add_subplot(grid5[2, 1])
    ax = a.plot_continuation('PAR(16)', 'PAR(20)', cont='c3:delta_p/k_p', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\Delta_p$')
    ax.set_ylabel(r'$k_p$')

    plt.tight_layout()

    plt.show()
