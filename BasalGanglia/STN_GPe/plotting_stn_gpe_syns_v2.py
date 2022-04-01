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


################
# file loading #
################

fname = 'results/stn_gpe_syns_v2.pkl'
a = PyAuto.from_file(fname)

##########################################
# investigation of GPe internal coupling #
##########################################

# continuation of GPe afferents
###############################

fig1 = plt.figure(tight_layout=True, figsize=(8.0, 4.0), dpi=dpi)
grid1 = gs.GridSpec(2, 2)

# codim 1 bifurcations
ax = fig1.add_subplot(grid1[0, :])
ax = a.plot_continuation('PAR(21)', 'U(3)', cont='k_gp', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E', default_size=markersize1)
ax = a.plot_continuation('PAR(21)', 'U(3)', cont='k_gp_lc', ax=ax, ignore=['BP'], line_color_stable='#148F77',
                         default_size=markersize1, custom_bf_styles={'LP': {'marker': 'p'}})
ax.set_xlabel(r'GPe afferents')
ax.set_ylabel('Firing rate (GPe-p)')
ax.set_title(r'Bursting Limit Cycle')

# codim 2 bifurcations
ax = fig1.add_subplot(grid1[1, 0])
ax = a.plot_continuation('PAR(21)', 'PAR(23)', cont='k_gp/k_gp_intra', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E', default_size=markersize1)
ax.set_xlabel(r'$k_{gp}$')
ax.set_ylabel(r'$k_{gp-intra}$')
ax = fig1.add_subplot(grid1[1, 1])
ax = a.plot_continuation('PAR(21)', 'PAR(24)', cont='k_gp/k_gp_inh', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E', default_size=markersize1)
ax.set_xlabel(r'$k_{gp}$')
ax.set_ylabel(r'$k_{gp-inh}$')
plt.tight_layout()
plt.show()

# continuation of GPe-p <-> GPe-a coupling
##########################################

fig2 = plt.figure(tight_layout=True, figsize=(8.0, 6.0), dpi=dpi)
grid2 = gs.GridSpec(3, 2)

# codim 1 bifurcations
ax = fig2.add_subplot(grid2[0, :])
ax = a.plot_continuation('PAR(23)', 'U(3)', cont='k_gp_intra', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E', default_size=markersize1)
ax = a.plot_continuation('PAR(23)', 'U(3)', cont='k_gp_intra_lc', ax=ax, ignore=['BP'], line_color_stable='#148F77',
                         default_size=markersize1, custom_bf_styles={'LP': {'marker': 'p'}})
ax.set_xlabel(r'$k_{gp-intra}$')
ax.set_ylabel('Firing rate (GPe-p)')
ax.set_title(r'Bursting Limit Cycle')

# codim 2 bifurcations
ax = fig2.add_subplot(grid2[1, 0])
ax = a.plot_continuation('PAR(23)', 'PAR(9)', cont='k_gp_intra/k_ap', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E', default_size=markersize1)
ax.set_xlabel(r'$k_{gp-intra}$')
ax.set_ylabel(r'$k_{ap}$')
ax = fig2.add_subplot(grid2[1, 1])
ax = a.plot_continuation('PAR(23)', 'PAR(10)', cont='k_gp_intra/k_pa', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E', default_size=markersize1)
ax.set_xlabel(r'$k_{gp-intra}$')
ax.set_ylabel(r'$k_{pa}$')
ax = fig2.add_subplot(grid2[2, 0])
ax = a.plot_continuation('PAR(23)', 'PAR(15)', cont='k_gp_intra/k_aa', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E', default_size=markersize1)
ax.set_xlabel(r'$k_{gp-intra}$')
ax.set_ylabel(r'$k_{aa}$')
ax = fig2.add_subplot(grid2[2, 1])
ax = a.plot_continuation('PAR(23)', 'PAR(8)', cont='k_gp_intra/k_pp', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E', default_size=markersize1)
ax.set_xlabel(r'$k_{gp-intra}$')
ax.set_ylabel(r'$k_{pp}$')

plt.tight_layout()

# continuation of striatal efferents
####################################

fig3 = plt.figure(tight_layout=True, figsize=(8.0, 7.0), dpi=dpi)
grid3 = gs.GridSpec(3, 2)

# codim 1 bifurcations
ax = fig3.add_subplot(grid3[0, :])
ax = a.plot_continuation('PAR(22)', 'U(3)', cont='str', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E', default_size=markersize1)
ax.set_xlabel(r'STR')
ax.set_ylabel('Firing rate (GPe-p)')
ax.set_title(r'Bursting Limit Cycle')
ax = fig3.add_subplot(grid3[1, :])
ax = a.plot_continuation('PAR(17)', 'U(3)', cont='k_as', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E', default_size=markersize1)
ax.set_xlabel(r'$k_{as}$')
ax.set_ylabel('Firing rate (GPe-p)')
ax.set_title(r'Bursting Limit Cycle')
ax = fig3.add_subplot(grid3[2, :])
ax = a.plot_continuation('PAR(16)', 'U(3)', cont='k_ps', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E', default_size=markersize1)
ax = a.plot_continuation('PAR(16)', 'U(3)', cont='k_ps_lc', ax=ax, ignore=['BP'], line_color_stable='#148F77',
                         default_size=markersize1, custom_bf_styles={'LP': {'marker': 'p'}})
ax.set_xlabel(r'$k_{ps}$')
ax.set_ylabel('Firing rate (GPe-p)')
ax.set_title(r'Bursting Limit Cycle')

plt.tight_layout()

###########################################
# investigation of firing rate statistics #
###########################################

# continuation of eta_e
#######################

fig4 = plt.figure(tight_layout=True, figsize=(8.0, 4.0), dpi=dpi)
grid4 = gs.GridSpec(1, 2)

# codim 1 bifurcations
ax = fig4.add_subplot(grid4[0, :])
ax = a.plot_continuation('PAR(1)', 'U(3)', cont='eta_e', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E', default_size=markersize1)
ax = a.plot_continuation('PAR(1)', 'U(3)', cont='eta_e_lc', ax=ax, ignore=['BP'], line_color_stable='#148F77',
                         default_size=markersize1, custom_bf_styles={'LP': {'marker': 'p'}})
ax.set_xlabel(r'$\eta_e$')
ax.set_ylabel('Firing rate (GPe-p)')
ax.set_title(r'Bursting Limit Cycle')

plt.tight_layout()

# continuation of delta_a
#########################

fig5 = plt.figure(tight_layout=True, figsize=(8.0, 4.0), dpi=dpi)
grid5 = gs.GridSpec(2, 2)

# codim 1 bifurcations
ax = fig5.add_subplot(grid5[0, :])
ax = a.plot_continuation('PAR(20)', 'U(3)', cont='delta_a', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E', default_size=markersize1)
ax = a.plot_continuation('PAR(20)', 'U(3)', cont='delta_a_lc', ax=ax, ignore=['BP'], line_color_stable='#148F77',
                         default_size=markersize1, custom_bf_styles={'LP': {'marker': 'p'}})
ax.set_xlabel(r'$\Delta_a$')
ax.set_ylabel('Firing rate (GPe-p)')
ax.set_title(r'Bursting Limit Cycle')

# codim 2 bifurcations
ax = fig5.add_subplot(grid5[1, :])
ax = a.plot_continuation('PAR(20)', 'PAR(19)', cont='delta_a/delta_p', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E', default_size=markersize1)
ax.set_xlabel(r'$\Delta_a$')
ax.set_ylabel(r'$Delta_p$')

plt.tight_layout()

plt.show()
