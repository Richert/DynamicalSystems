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
markersize2 = 40
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

c1 = True  # strong GPe-p projections
c2 = False  # strong bidirectional coupling between GPe-p and GPe-a
c3 = False  # weak bidirectional coupling between GPe-p and GPe-a

##########################################################################
# c1: investigation of GPe behavior for strong GPe-p to GPe-a connection #
##########################################################################

if c1:

    # continuation of eta_a
    #######################

    fig1 = plt.figure(tight_layout=True, figsize=(6.0, 9.0), dpi=dpi)
    grid1 = gs.GridSpec(3, 2)

    # codim 1
    ax = fig1.add_subplot(grid1[0, :])
    ax = a.plot_continuation('PAR(7)', 'U(2)', cont='c1:k_ap', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax = a.plot_continuation('PAR(7)', 'U(2)', cont='c1:k_ap_lc', ax=ax, ignore=['BP'], line_color_stable='#148F77',
                             default_size=markersize1, custom_bf_styles={'LP': {'marker': 'p'}})
    ax.set_xlabel(r'$k_{ap}$')
    ax.set_ylabel('Firing rate')

    # codim 2
    ax = fig1.add_subplot(grid1[1, 0])
    ax = a.plot_continuation('PAR(7)', 'PAR(8)', cont='c1:k_ap/k_pa', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize2)
    ax.set_xlabel(r'$k_{ap}$')
    ax.set_ylabel(r'$k_{pa}$')

    ax = fig1.add_subplot(grid1[1, 1])
    ax = a.plot_continuation('PAR(20)', 'PAR(21)', cont='c1:k_p/k_i', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize2)
    ax.set_xlabel(r'$k_p$')
    ax.set_ylabel(r'$k_i$')

    ax = fig1.add_subplot(grid1[2, 0])
    ax = a.plot_continuation('PAR(2)', 'PAR(21)', cont='c1:eta_p/k_i', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize2)
    ax.set_xlabel(r'$\eta_p$')
    ax.set_ylabel(r'$k_i$')

    ax = fig1.add_subplot(grid1[2, 1])
    ax = a.plot_continuation('PAR(2)', 'PAR(20)', cont='c1:eta_p/k_p', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize2)
    ax.set_xlabel(r'$\eta_p$')
    ax.set_ylabel(r'$k_p$')

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
                             line_color_unstable='#5D6D7E', default_size=markersize1, ignore=['UZ'])
    ax = a.plot_continuation('PAR(2)', 'U(4)', cont='c2:eta_p', ax=ax, line_color_stable='#8299b0',
                             line_color_unstable='#5D6D7E', default_size=markersize1, ignore=['UZ'])
    ax.set_xlabel(r'$\eta_p$')
    ax.set_ylabel('Firing rate')
    ax.set_xlim([0.9, 3.1])
    ax.set_ylim([0.0, 0.13])
    ax.set_yticklabels([0.0, 50.0, 100.0])

    # codim 2
    ax = fig3.add_subplot(grid3[1, 0])
    ax = a.plot_continuation('PAR(20)', 'PAR(3)', cont='c2:k_p/eta_a', ax=ax, line_color_stable='#5D6D7E',
                             line_color_unstable='#5D6D7E', default_size=markersize1, ignore=['LP'])
    ax = a.plot_continuation('PAR(20)', 'PAR(3)', cont='c2:k_p/eta_a/zh1', ax=ax, line_color_stable='#3689c9',
                             line_color_unstable='#3689c9', default_size=markersize1, ignore=['LP'])
    ax.set_xlabel(r'$k_p$')
    ax.set_ylabel(r'$\eta_a$')
    #ax.set_ylim([-5.0, 20.0])
    #ax.set_xlim([-2.0, 20.0])

    ax = fig3.add_subplot(grid3[1, 1])
    ax = a.plot_continuation('PAR(2)', 'PAR(20)', cont='c2:eta_p/k_p', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1, ignore=['LP'])
    ax.set_xlabel(r'$\eta_p$')
    ax.set_ylabel(r'$k_p$')
    ax.set_ylim([-0.1, 1.3])

    ax = fig3.add_subplot(grid3[2, 0])
    ax = a.plot_continuation('PAR(20)', 'PAR(21)', cont='c2:k_p/k_i', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1, ignore=['LP'])
    ax.set_xlabel(r'$k_{p}$')
    ax.set_ylabel(r'$k_{i}$')
    ax.set_ylim([0, 10])

    ax = fig3.add_subplot(grid3[2, 1])
    ax = a.plot_continuation('PAR(2)', 'PAR(21)', cont='c2:eta_p/k_i', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1, ignore=['LP'])
    ax.set_xlabel(r'$\eta_p$')
    ax.set_ylabel(r'$k_{i}$')
    ax.set_ylim([0, 10])

    plt.tight_layout()

    plt.show()

#######################################################################
# c3: investigation of GPe behavior for weak GPe-p <-> GPe-a coupling #
#######################################################################

if c3:

    # continuation of delta_a
    #########################

    fig4 = plt.figure(tight_layout=True, figsize=(6.0, 9.0), dpi=dpi)
    grid4 = gs.GridSpec(3, 2)

    # codim 1
    ax = fig4.add_subplot(grid4[0, :])
    ax = a.plot_continuation('PAR(17)', 'U(2)', cont='c3:delta_a', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax = a.plot_continuation('PAR(17)', 'U(2)', cont='c3:delta_a_lc', ax=ax, ignore=['BP'], line_color_stable='#148F77',
                             default_size=markersize1, custom_bf_styles={'LP': {'marker': 'p'}})
    ax = a.plot_continuation('PAR(17)', 'U(2)', cont='c3:delta_a_lc2', ax=ax, ignore=['BP'], line_color_stable='#148F77',
                             default_size=markersize1, custom_bf_styles={'LP': {'marker': 'p'}})
    ax.set_xlabel(r'$\Delta_a$')
    ax.set_ylabel('Firing rate')
    #ax.set_xlim([0.0, 0.11])
    #ax.set_yticklabels([0.0, 200.0, 400.0])

    # codim 2
    ax = fig4.add_subplot(grid4[1, 0])
    ax = a.plot_continuation('PAR(17)', 'PAR(20)', cont='c3:delta_a/k_p', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\Delta_a$')
    ax.set_ylabel(r'$k_p$')

    ax = fig4.add_subplot(grid4[1, 1])
    ax = a.plot_continuation('PAR(17)', 'PAR(21)', cont='c3:delta_a/k_i', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\Delta_a$')
    ax.set_ylabel(r'$k_i$')

    ax = fig4.add_subplot(grid4[2, 0])
    ax = a.plot_continuation('PAR(7)', 'PAR(8)', cont='c3:k_pa/k_ap/delta_a', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$k_{ap}$')
    ax.set_ylabel(r'$k_{pa}$')

    ax = fig4.add_subplot(grid4[2, 1])
    ax = a.plot_continuation('PAR(21)', 'PAR(22)', cont='c3:k_i/k_pi/delta_a', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$k_i$')
    ax.set_ylabel(r'$k_{pi}$')

    plt.tight_layout()

    # continuation of eta_a
    #######################

    fig5 = plt.figure(tight_layout=True, figsize=(6.0, 9.0), dpi=dpi)
    grid5 = gs.GridSpec(3, 2)

    # codim 1
    ax = fig5.add_subplot(grid5[0, :])
    ax = a.plot_continuation('PAR(3)', 'U(2)', cont='c3:eta_a', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax = a.plot_continuation('PAR(3)', 'U(2)', cont='c3:eta_a_lc', ax=ax, ignore=['BP'], line_color_stable='#148F77',
                             default_size=markersize1, custom_bf_styles={'LP': {'marker': 'p'}})
    ax = a.plot_continuation('PAR(3)', 'U(2)', cont='c3:eta_a_lc2', ax=ax, ignore=['BP'], line_color_stable='#148F77',
                             default_size=markersize1, custom_bf_styles={'LP': {'marker': 'p'}})
    ax.set_xlabel(r'$\eta_a$')
    ax.set_ylabel('Firing rate')

    # codim 2
    ax = fig5.add_subplot(grid5[1, 0])
    ax = a.plot_continuation('PAR(3)', 'PAR(2)', cont='c3:eta_a/eta_p', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\eta_a$')
    ax.set_ylabel(r'$\eta_p$')

    ax = fig5.add_subplot(grid5[1, 1])
    ax = a.plot_continuation('PAR(8)', 'PAR(7)', cont='c3:k_pa/k_ap', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$k_{ap}$')
    ax.set_ylabel(r'$k_{pa}$')

    ax = fig5.add_subplot(grid5[2, 0])
    ax = a.plot_continuation('PAR(3)', 'PAR(20)', cont='c3:eta_a/k_p', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\eta_a$')
    ax.set_ylabel(r'$k_p$')

    ax = fig5.add_subplot(grid5[2, 1])
    ax = a.plot_continuation('PAR(3)', 'PAR(21)', cont='c3:eta_a/k_i', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax.set_xlabel(r'$\eta_a$')
    ax.set_ylabel(r'$k_i$')

    plt.tight_layout()

    plt.show()
