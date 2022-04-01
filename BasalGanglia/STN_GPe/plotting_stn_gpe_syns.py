import matplotlib.pyplot as plt
import matplotlib as mpl
from pyrates.utility.pyauto import PyAuto
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

c1 = [  # weak GPe-p <-> GPe-a coupling
      False,  # STN -> GPe-p < STN -> GPe-a
      False,   # STN -> GPe-p > STN -> GPe-a
]
c2 = [  # balanced GPe-p <-> GPe-a coupling
      False,  # STN -> GPe-p < STN -> GPe-a
      False,   # STN -> GPe-p > STN -> GPe-a
]
c3 = [  # strong GPe-p <-> GPe-a coupling
      False,  # STN -> GPe-p < STN -> GPe-a
      True,   # STN -> GPe-p > STN -> GPe-a
]

########################################################################
# c1: investigation of GPe behavior for weak inter-population coupling #
########################################################################

if any(c1):

    if c1[0]:

        # continuation of k_ap
        ######################

        fig1 = plt.figure(tight_layout=True, figsize=(6.0, 9.0), dpi=dpi)
        grid1 = gs.GridSpec(3, 2)

        # codim 1: eta_e
        ax = fig1.add_subplot(grid1[0, :])
        ax = a.plot_continuation('PAR(9)', 'U(3)', cont='c1:k_ap', ax=ax, line_color_stable='#76448A',
                                 line_color_unstable='#5D6D7E', default_size=markersize1)
        ax = a.plot_continuation('PAR(9)', 'U(3)', cont='c1:k_ap_lc', ax=ax, ignore=['BP'],
                                 line_color_stable='#148F77', default_size=markersize1)
        ax.set_xlabel(r'$k_{ap}$')
        ax.set_ylabel('Firing rate')
        # ax.set_xlim([-6.5, 9.0])
        # ax.set_ylim([0.0, 0.13])
        # ax.set_yticks([0.0, 0.025, 0.05, 0.075, 0.1, 0.125])
        # ax.set_yticklabels([0.0, 25.0, 50.0, 75.0, 100.0, 125.0])

        # codim 2: eta_p and k_i
        ax = fig1.add_subplot(grid1[1, 0])
        ax = a.plot_continuation('PAR(9)', 'PAR(24)', cont=f'c1:k_i/k_ap', ax=ax, line_color_stable='#5D6D7E',
                                 line_color_unstable='#5D6D7E', default_size=markersize1)
        ax.set_xlabel(r'$k_{ap}$')
        ax.set_ylabel(r'$k_{i}$')
        #ax.set_xlim([-12.0, 12.0])
        #ax.set_ylim([0.0, 3.0])

        # codim 2: eta_a and k_ap
        ax = fig1.add_subplot(grid1[1, 1])
        ax = a.plot_continuation('PAR(9)', 'PAR(26)', cont=f'c1:k_gpe_e/k_ap', ax=ax, line_color_stable='#5D6D7E',
                                 line_color_unstable='#5D6D7E', default_size=markersize1)
        ax.set_xlabel(r'$k_{ap}$')
        ax.set_ylabel(r'$k_{stn}$')
        #ax.set_xlim([0.0, 10.0])
        #ax.set_ylim([0.0, 3.0])
        # ax.set_xticks([0.0, 5.0, 10.0])

        ax = fig1.add_subplot(grid1[2, 0])
        ax = a.plot_continuation('PAR(9)', 'PAR(8)', cont=f'c1:k_pp/k_ap', ax=ax, line_color_stable='#5D6D7E',
                                 line_color_unstable='#5D6D7E', default_size=markersize1)
        ax.set_xlabel(r'$k_{ap}$')
        ax.set_ylabel(r'$k_{pp}$')
        # ax.set_xlim([0.0, 10.0])
        # ax.set_ylim([0.0, 3.0])

        ax = fig1.add_subplot(grid1[2, 1])
        ax = a.plot_continuation('PAR(9)', 'PAR(22)', cont=f'c1:k_gp/k_ap', ax=ax, line_color_stable='#5D6D7E',
                                 line_color_unstable='#5D6D7E', default_size=markersize1)
        ax.set_xlabel(r'$k_{ap}$')
        ax.set_ylabel(r'$k_{gp}$')
        #ax.set_xlim([0.0, 10.0])
        #ax.set_ylim([0.0, 3.0])

        plt.tight_layout()

        plt.show()

    if c1[1]:

        # continuation of k_ap
        ######################

        fig2 = plt.figure(tight_layout=True, figsize=(6.0, 9.0), dpi=dpi)
        grid2 = gs.GridSpec(3, 2)

        # codim 1: eta_e
        ax = fig2.add_subplot(grid2[0, :])
        ax = a.plot_continuation('PAR(9)', 'U(3)', cont='c1.2:k_ap', ax=ax, line_color_stable='#76448A',
                                 line_color_unstable='#5D6D7E', default_size=markersize1)
        ax = a.plot_continuation('PAR(9)', 'U(3)', cont='c1.2:k_ap_lc', ax=ax, ignore=['BP'],
                                 line_color_stable='#148F77', default_size=markersize1)
        ax.set_xlabel(r'$k_{ap}$')
        ax.set_ylabel('Firing rate')
        # ax.set_xlim([-6.5, 9.0])
        # ax.set_ylim([0.0, 0.13])
        # ax.set_yticks([0.0, 0.025, 0.05, 0.075, 0.1, 0.125])
        # ax.set_yticklabels([0.0, 25.0, 50.0, 75.0, 100.0, 125.0])

        # codim 2: eta_p and k_i
        ax = fig2.add_subplot(grid2[1, 0])
        ax = a.plot_continuation('PAR(9)', 'PAR(24)', cont=f'c1.2:k_i/k_ap', ax=ax, line_color_stable='#5D6D7E',
                                 line_color_unstable='#5D6D7E', default_size=markersize1)
        ax.set_xlabel(r'$k_{ap}$')
        ax.set_ylabel(r'$k_{i}$')
        # ax.set_xlim([-12.0, 12.0])
        # ax.set_ylim([0.0, 3.0])

        # codim 2: eta_a and k_ap
        ax = fig2.add_subplot(grid2[1, 1])
        ax = a.plot_continuation('PAR(9)', 'PAR(26)', cont=f'c1.2:k_gpe_e/k_ap', ax=ax, line_color_stable='#5D6D7E',
                                 line_color_unstable='#5D6D7E', default_size=markersize1)
        ax.set_xlabel(r'$k_{ap}$')
        ax.set_ylabel(r'$k_{stn}$')
        # ax.set_xlim([0.0, 10.0])
        # ax.set_ylim([0.0, 3.0])
        # ax.set_xticks([0.0, 5.0, 10.0])

        ax = fig2.add_subplot(grid2[2, 0])
        ax = a.plot_continuation('PAR(9)', 'PAR(8)', cont=f'c1.2:k_pp/k_ap', ax=ax, line_color_stable='#5D6D7E',
                                 line_color_unstable='#5D6D7E', default_size=markersize1)
        ax.set_xlabel(r'$k_{ap}$')
        ax.set_ylabel(r'$k_{pp}$')
        # ax.set_xlim([0.0, 10.0])
        # ax.set_ylim([0.0, 3.0])

        ax = fig2.add_subplot(grid2[2, 1])
        ax = a.plot_continuation('PAR(9)', 'PAR(22)', cont=f'c1.2:k_gp/k_ap', ax=ax, line_color_stable='#5D6D7E',
                                 line_color_unstable='#5D6D7E', default_size=markersize1)
        ax.set_xlabel(r'$k_{ap}$')
        ax.set_ylabel(r'$k_{gp}$')
        # ax.set_xlim([0.0, 10.0])
        # ax.set_ylim([0.0, 3.0])

        plt.tight_layout()

        plt.show()

############################################################################
# c2: investigation of GPe behavior for balanced inter-population coupling #
############################################################################

if any(c2):

    if c2[0]:

        # continuation of k_ap
        ######################

        fig1 = plt.figure(tight_layout=True, figsize=(6.0, 9.0), dpi=dpi)
        grid1 = gs.GridSpec(3, 2)

        # codim 1: eta_e
        ax = fig1.add_subplot(grid1[0, :])
        ax = a.plot_continuation('PAR(9)', 'U(3)', cont='c1:k_ap', ax=ax, line_color_stable='#76448A',
                                 line_color_unstable='#5D6D7E', default_size=markersize1)
        ax = a.plot_continuation('PAR(9)', 'U(3)', cont='c1:k_ap_lc', ax=ax, ignore=['BP'],
                                 line_color_stable='#148F77', default_size=markersize1)
        ax.set_xlabel(r'$k_{ap}$')
        ax.set_ylabel('Firing rate')
        # ax.set_xlim([-6.5, 9.0])
        # ax.set_ylim([0.0, 0.13])
        # ax.set_yticks([0.0, 0.025, 0.05, 0.075, 0.1, 0.125])
        # ax.set_yticklabels([0.0, 25.0, 50.0, 75.0, 100.0, 125.0])

        # codim 2: eta_p and k_i
        ax = fig1.add_subplot(grid1[1, 0])
        ax = a.plot_continuation('PAR(9)', 'PAR(24)', cont=f'c1:k_i/k_ap', ax=ax, line_color_stable='#5D6D7E',
                                 line_color_unstable='#5D6D7E', default_size=markersize1)
        ax.set_xlabel(r'$k_{ap}$')
        ax.set_ylabel(r'$k_{i}$')
        # ax.set_xlim([-12.0, 12.0])
        # ax.set_ylim([0.0, 3.0])

        # codim 2: eta_a and k_ap
        ax = fig1.add_subplot(grid1[1, 1])
        ax = a.plot_continuation('PAR(9)', 'PAR(26)', cont=f'c1:k_gpe_e/k_ap', ax=ax, line_color_stable='#5D6D7E',
                                 line_color_unstable='#5D6D7E', default_size=markersize1)
        ax.set_xlabel(r'$k_{ap}$')
        ax.set_ylabel(r'$k_{stn}$')
        # ax.set_xlim([0.0, 10.0])
        # ax.set_ylim([0.0, 3.0])
        # ax.set_xticks([0.0, 5.0, 10.0])

        ax = fig1.add_subplot(grid1[2, 0])
        ax = a.plot_continuation('PAR(9)', 'PAR(8)', cont=f'c1:k_pp/k_ap', ax=ax, line_color_stable='#5D6D7E',
                                 line_color_unstable='#5D6D7E', default_size=markersize1)
        ax.set_xlabel(r'$k_{ap}$')
        ax.set_ylabel(r'$k_{pp}$')
        # ax.set_xlim([0.0, 10.0])
        # ax.set_ylim([0.0, 3.0])

        ax = fig1.add_subplot(grid1[2, 1])
        ax = a.plot_continuation('PAR(9)', 'PAR(22)', cont=f'c1:k_gp/k_ap', ax=ax, line_color_stable='#5D6D7E',
                                 line_color_unstable='#5D6D7E', default_size=markersize1)
        ax.set_xlabel(r'$k_{ap}$')
        ax.set_ylabel(r'$k_{gp}$')
        # ax.set_xlim([0.0, 10.0])
        # ax.set_ylim([0.0, 3.0])

        plt.tight_layout()

        plt.show()

    if c2[1]:

        # continuation of k_gp
        ######################

        fig2 = plt.figure(tight_layout=True, figsize=(6.0, 9.0), dpi=dpi)
        grid2 = gs.GridSpec(3, 2)

        # codim 1: eta_e
        ax = fig2.add_subplot(grid2[0, :])
        ax = a.plot_continuation('PAR(22)', 'U(3)', cont='c2.2:k_gp', ax=ax, line_color_stable='#76448A',
                                 line_color_unstable='#5D6D7E', default_size=markersize1)
        ax = a.plot_continuation('PAR(22)', 'U(3)', cont='c2.2:k_ap_lc', ax=ax, ignore=['BP'],
                                 line_color_stable='#148F77', default_size=markersize1)
        ax = a.plot_continuation('PAR(22)', 'U(3)', cont='c2.2:k_ap_lc2', ax=ax, ignore=['BP'],
                                 line_color_stable='#148F77', default_size=markersize1)
        ax.set_xlabel(r'$k_{gp}$')
        ax.set_ylabel('Firing rate')
        # ax.set_xlim([-6.5, 9.0])
        # ax.set_ylim([0.0, 0.13])
        # ax.set_yticks([0.0, 0.025, 0.05, 0.075, 0.1, 0.125])
        # ax.set_yticklabels([0.0, 25.0, 50.0, 75.0, 100.0, 125.0])

        # codim 2: eta_p and k_i
        ax = fig2.add_subplot(grid2[1, 0])
        ax = a.plot_continuation('PAR(22)', 'PAR(24)', cont=f'c2.2:k_gp/k_i', ax=ax, line_color_stable='#5D6D7E',
                                 line_color_unstable='#5D6D7E', default_size=markersize1)
        ax.set_xlabel(r'$k_{gp}$')
        ax.set_ylabel(r'$k_{i}$')
        # ax.set_xlim([-12.0, 12.0])
        # ax.set_ylim([0.0, 3.0])

        # codim 2: eta_a and k_ap
        ax = fig2.add_subplot(grid2[1, 1])
        ax = a.plot_continuation('PAR(22)', 'PAR(26)', cont=f'c2.2:k_gp/k_gp_e', ax=ax, line_color_stable='#5D6D7E',
                                 line_color_unstable='#5D6D7E', default_size=markersize1)
        ax.set_xlabel(r'$k_{gp}$')
        ax.set_ylabel(r'$k_{stn}$')
        # ax.set_xlim([0.0, 10.0])
        # ax.set_ylim([0.0, 3.0])
        # ax.set_xticks([0.0, 5.0, 10.0])

        ax = fig2.add_subplot(grid2[2, 0])
        ax = a.plot_continuation('PAR(22)', 'PAR(10)', cont=f'c2.2:k_gp/k_pa', ax=ax, line_color_stable='#5D6D7E',
                                 line_color_unstable='#5D6D7E', default_size=markersize1)
        ax.set_xlabel(r'$k_{ap}$')
        ax.set_ylabel(r'$k_{pp}$')
        # ax.set_xlim([0.0, 10.0])
        # ax.set_ylim([0.0, 3.0])

        ax = fig2.add_subplot(grid2[2, 1])
        ax = a.plot_continuation('PAR(22)', 'PAR(8)', cont=f'c2.2:k_gp/k_pp', ax=ax, line_color_stable='#5D6D7E',
                                 line_color_unstable='#5D6D7E', default_size=markersize1)
        ax.set_xlabel(r'$k_{ap}$')
        ax.set_ylabel(r'$k_{gp}$')
        # ax.set_xlim([0.0, 10.0])
        # ax.set_ylim([0.0, 3.0])

        plt.tight_layout()

        plt.show()

#########################################################################
# c3: investigation of GPe behavior for strong GPe-p <-> GPe-a coupling #
#########################################################################

if any(c3):

    if c3[1]:

        branch_color = {'HB': '#8299b0',
                        'LP': '#b8b632'}

        # 2D continuation k_gp x k_gpe_e
        ################################

        fig1 = plt.figure(tight_layout=True, figsize=(6.0, 6.0), dpi=dpi)
        grid1 = gs.GridSpec(2, 2)
        ax = fig1.add_subplot(grid1[:, :])
        for cont in a.additional_attributes['k_gp/k_gp_e:names']:
            if "(LP)" in cont:
                color = '#8299b0'
            else:
                color = '#3689c9'
            ax = a.plot_continuation('PAR(22)', 'PAR(26)', cont=cont, ax=ax, line_color_stable=color,
                                     line_color_unstable=color, default_size=markersize1,
                                     line_style_unstable='solid', ignore=['LP'])
        ax.set_xlabel(r'$k_{gp}$')
        ax.set_ylabel(r'$k_{stn}$')
        plt.tight_layout()

        # 2D continuation k_gp x k_i
        ############################

        fig2 = plt.figure(tight_layout=True, figsize=(6.0, 6.0), dpi=dpi)
        grid2 = gs.GridSpec(2, 2)
        ax = fig2.add_subplot(grid2[:, :])
        for cont in a.additional_attributes['k_gp/eta_e:names']:
            if "(LP)" in cont:
                color = '#8299b0'
            else:
                color = '#3689c9'
            ax = a.plot_continuation('PAR(22)', 'PAR(1)', cont=cont, ax=ax, line_color_stable=color,
                                     line_color_unstable=color, default_size=markersize1,
                                     line_style_unstable='solid', ignore=['LP'])
        ax.set_xlabel(r'$k_{gp}$')
        ax.set_ylabel(r'$\eta_p$')
        plt.tight_layout()

        # 2D continuation k_i x k_gp_e
        ##############################

        fig3 = plt.figure(tight_layout=True, figsize=(6.0, 6.0), dpi=dpi)
        grid3 = gs.GridSpec(2, 2)
        ax = fig3.add_subplot(grid3[:, :])
        for cont in a.additional_attributes['k_i/k_gp_e:names']:
            if "(LP)" in cont:
                color = '#8299b0'
            else:
                color = '#3689c9'
            ax = a.plot_continuation('PAR(24)', 'PAR(26)', cont=cont, ax=ax, line_color_stable=color,
                                     line_color_unstable=color, default_size=markersize1,
                                     line_style_unstable='solid', ignore=['LP'])
        ax.set_xlabel(r'$k_{i}$')
        ax.set_ylabel(r'$k_{stn}$')
        plt.tight_layout()
        plt.show()
