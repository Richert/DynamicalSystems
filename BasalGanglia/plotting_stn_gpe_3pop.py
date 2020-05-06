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

fname = 'results/stn_gpe_3pop.pkl'
a = PyAuto.from_file(fname)

############
# plotting #
############

# principle continuation in dopa
################################

fig1 = plt.figure(tight_layout=True, figsize=(8.0, 6.0), dpi=dpi)
grid1 = gs.GridSpec(3, 2)

# codim 1 bifurcations
ax = fig1.add_subplot(grid1[0, :])
ax = a.plot_continuation('PAR(1)', 'U(2)', cont='dopa', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E', default_size=markersize1)
ax = a.plot_continuation('PAR(1)', 'U(2)', cont='dopa_lc', ax=ax, ignore=['BP'], line_color_stable='#148F77',
                         default_size=markersize1, custom_bf_styles={'LP': {'marker': 'p'}})
ax.set_xlabel(r'd')
ax.set_ylabel('Firing rate (STN)')
ax.set_title(r'Bursting Limit Cycle')

# codim 2 bifurcations
ax = fig1.add_subplot(grid1[1, 0])
ax = a.plot_continuation('PAR(1)', 'PAR(16)', cont='dopa/delta_p', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E', default_size=markersize1)
ax.set_xlabel(r'd')
ax.set_ylabel(r'$\Delta_p$')
ax = fig1.add_subplot(grid1[1, 1])
ax = a.plot_continuation('PAR(1)', 'PAR(17)', cont='dopa/delta_a', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E', default_size=markersize1)
ax.set_xlabel(r'd')
ax.set_ylabel(r'$\Delta_a$')
ax = fig1.add_subplot(grid1[2, 0])
ax = a.plot_continuation('PAR(1)', 'PAR(8)', cont='dopa/k_ap', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E', default_size=markersize1)
ax.set_xlabel(r'd')
ax.set_ylabel(r'$k_{ap}$')
ax = fig1.add_subplot(grid1[2, 1])
ax = a.plot_continuation('PAR(1)', 'PAR(5)', cont='dopa/k_pa', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E', default_size=markersize1)
ax.set_xlabel(r'd')
ax.set_ylabel(r'$k_{pa}$')
plt.tight_layout()

# continuation in delta_p
#########################

fig2 = plt.figure(tight_layout=True, figsize=(6.0, 3.0), dpi=dpi)
grid2 = gs.GridSpec(1, 2)

# plot eta continuation for single alpha with limit cycle continuation
ax = fig2.add_subplot(grid2[0, :])
ax = a.plot_continuation('PAR(16)', 'U(2)', cont='delta_p', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E', default_size=markersize1)
ax = a.plot_continuation('PAR(16)', 'U(2)', cont='delta_p_lc', ax=ax, ignore=['BP'], line_color_stable='#148F77',
                         default_size=markersize1, custom_bf_styles={'LP': {'marker': 'p'}})
ax.set_xlabel(r'$\Delta_p$')
ax.set_ylabel('Firing rate (STN)')
ax.set_title(r'Bursting Limit Cycle')

plt.tight_layout()

# continuation in k_ae
######################

fig3 = plt.figure(tight_layout=True, figsize=(6.0, 3.0), dpi=dpi)
grid3 = gs.GridSpec(1, 2)

# plot eta continuation for single alpha with limit cycle continuation
ax = fig3.add_subplot(grid3[0, :])
ax = a.plot_continuation('PAR(7)', 'U(2)', cont='k_ae', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E', default_size=markersize1)
ax = a.plot_continuation('PAR(7)', 'U(2)', cont='k_ae_lc', ax=ax, ignore=['BP'], line_color_stable='#148F77',
                         default_size=markersize1, custom_bf_styles={'LP': {'marker': 'p'}})
ax.set_xlabel(r'$k_{ae}$')
ax.set_ylabel('Firing rate (STN)')
ax.set_title(r'Bursting Limit Cycle')

plt.tight_layout()

# continuation in k_p
#####################

fig4 = plt.figure(tight_layout=True, figsize=(6.0, 3.0), dpi=dpi)
grid4 = gs.GridSpec(1, 2)

# plot eta continuation for single alpha with limit cycle continuation
ax = fig4.add_subplot(grid4[0, :])
ax = a.plot_continuation('PAR(25)', 'U(2)', cont='k_p', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E', default_size=markersize1)
ax = a.plot_continuation('PAR(25)', 'U(2)', cont='k_p_lc', ax=ax, ignore=['BP'], line_color_stable='#148F77',
                         default_size=markersize1, custom_bf_styles={'LP': {'marker': 'p'}})
ax.set_xlabel(r'$k_{p}$')
ax.set_ylabel('Firing rate (STN)')
ax.set_title(r'Bursting Limit Cycle')

plt.tight_layout()

# continuation in k_gp
######################

fig5 = plt.figure(tight_layout=True, figsize=(6.0, 3.0), dpi=dpi)
grid5 = gs.GridSpec(1, 2)

# plot eta continuation for single alpha with limit cycle continuation
ax = fig5.add_subplot(grid5[0, :])
ax = a.plot_continuation('PAR(26)', 'U(2)', cont='k_gp', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E', default_size=markersize1)
ax = a.plot_continuation('PAR(26)', 'U(2)', cont='k_gp_lc', ax=ax, ignore=['BP'], line_color_stable='#148F77',
                         default_size=markersize1, custom_bf_styles={'LP': {'marker': 'p'}})
#ax = a.plot_continuation('PAR(26)', 'U(2)', cont='k_gp_lc2', ax=ax, ignore=['BP'], line_color_stable='#148F77',
#                         default_size=markersize1, custom_bf_styles={'LP': {'marker': 'p'}})
ax.set_xlabel(r'$k_{gp}$')
ax.set_ylabel('Firing rate (STN)')
ax.set_title(r'Bursting Limit Cycle')

plt.tight_layout()

# principle continuation in eta_e
#################################

fig6 = plt.figure(tight_layout=True, figsize=(8.0, 6.0), dpi=dpi)
grid6 = gs.GridSpec(3, 2)

# codim 1 bifurcations
ax = fig6.add_subplot(grid6[0, :])
ax = a.plot_continuation('PAR(18)', 'U(2)', cont='eta_e', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E', default_size=markersize1)
ax = a.plot_continuation('PAR(18)', 'U(2)', cont='eta_e_lc', ax=ax, ignore=['BP'], line_color_stable='#148F77',
                         default_size=markersize1, custom_bf_styles={'LP': {'marker': 'p'}})
#ax = a.plot_continuation('PAR(18)', 'U(2)', cont='eta_e_lc2', ax=ax, ignore=['BP'], line_color_stable='#148F77',
#                         default_size=markersize1, custom_bf_styles={'LP': {'marker': 'p'}})
ax.set_xlabel(r'$\eta_e$')
ax.set_ylabel('Firing rate (STN)')
ax.set_title(r'Bursting Limit Cycle')

# codim 2 bifurcations
ax = fig6.add_subplot(grid6[1, 0])
ax = a.plot_continuation('PAR(18)', 'PAR(2)', cont='eta_e/k_ep', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E', default_size=markersize1)
ax.set_xlabel(r'$\eta_e$')
ax.set_ylabel(r'$k_{ep}$')
ax = fig6.add_subplot(grid6[1, 1])
ax = a.plot_continuation('PAR(18)', 'PAR(16)', cont='eta_e/delta_e', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E', default_size=markersize1)
ax.set_xlabel(r'$\eta_e$')
ax.set_ylabel(r'$\Delta_{p}$')
ax = fig6.add_subplot(grid6[2, 0])
ax = a.plot_continuation('PAR(18)', 'PAR(19)', cont='eta_e/eta_p', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E', default_size=markersize1)
ax.set_xlabel(r'$\eta_e$')
ax.set_ylabel(r'$\eta_p$')
ax = fig6.add_subplot(grid6[2, 1])
ax = a.plot_continuation('PAR(18)', 'PAR(20)', cont='eta_e/eta_a', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E', default_size=markersize1)
ax.set_xlabel(r'$\eta_e$')
ax.set_ylabel(r'$\eta_a$')
plt.tight_layout()
plt.show()
