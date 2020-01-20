import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gs
from pyauto import PyAuto

# plotting parameters
linewidth = 2.0
linewidth2 = 2.0
fontsize1 = 12
fontsize2 = 14
markersize1 = 100
markersize2 = 100
dpi = 500

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

fname = 'results/qif_stn_gpe.pkl'
a = PyAuto.from_file(fname)

###################################
# plotting of qif_stn_gpe results #
###################################

# continuation in alpha
#######################

fig0 = plt.figure(tight_layout=True)
grid0 = gs.GridSpec(2, 3)

# codim 1
ax0 = fig0.add_subplot(grid0[0, :])
a.plot_continuation('PAR(17)', 'U(3)', cont='alpha', ax=ax0, line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E', ignore=['UZ'])
ax0.set_xlabel('alpha')
ax0.set_ylabel('r_i')

# codim 2
ax1 = fig0.add_subplot(grid0[1, 0])
a.plot_continuation('PAR(17)', 'PAR(3)', cont='eta_str/alpha', ax=ax1, line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
ax1.set_xlabel('alpha')
ax1.set_ylabel('eta_str')

ax2 = fig0.add_subplot(grid0[1, 1])
a.plot_continuation('PAR(17)', 'PAR(9)', cont='k/alpha', ax=ax2, line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
ax2.set_xlabel('alpha')
ax2.set_ylabel('k')

ax3 = fig0.add_subplot(grid0[1, 2])
a.plot_continuation('PAR(17)', 'PAR(10)', cont='k_i/alpha', ax=ax3, line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
ax3.set_xlabel('alpha')
ax3.set_ylabel('k_i')

plt.savefig('results/alpha_cont.svg', dpi=600, format='svg')

# continuation in eta_str
#########################

fig1 = plt.figure(tight_layout=True, figsize=(10, 10))
grid1 = gs.GridSpec(4, 3)

# codim 1
ax0 = fig1.add_subplot(grid1[0, :])
a.plot_continuation('PAR(3)', 'U(3)', cont='eta_str', ax=ax0, line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
a.plot_continuation('PAR(3)', 'U(3)', cont=f'eta_str_lc', ax=ax0, ignore=['BP'], line_color_stable='#148F77')
a.plot_continuation('PAR(3)', 'U(3)', cont=f'eta_str_lc2', ax=ax0, ignore=['BP'], line_color_stable='#148F77')
ax0.set_xlabel('eta_str')
ax0.set_ylabel('r_i')

# codim 2
ax1 = fig1.add_subplot(grid1[1, 0])
a.plot_continuation('PAR(3)', 'PAR(9)', cont='k/eta_str', ax=ax1, line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
ax1.set_xlabel('eta_str')
ax1.set_ylabel('k')

ax2 = fig1.add_subplot(grid1[1, 1])
a.plot_continuation('PAR(3)', 'PAR(17)', cont='alpha/eta_str', ax=ax2, line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
ax2.set_xlabel('eta_str')
ax2.set_ylabel('alpha')

ax3 = fig1.add_subplot(grid1[1, 2])
a.plot_continuation('PAR(3)', 'PAR(10)', cont='k_i/eta_str', ax=ax3, line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
ax3.set_xlabel('eta_str')
ax3.set_ylabel('k_i')

ax4 = fig1.add_subplot(grid1[2, 0])
a.plot_continuation('PAR(3)', 'PAR(9)', cont='k/eta_str_2', ax=ax4, line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E', ignore=['LP'])
ax4.set_xlabel('eta_str')
ax4.set_ylabel('k')

ax5 = fig1.add_subplot(grid1[2, 1])
a.plot_continuation('PAR(3)', 'PAR(17)', cont='alpha/eta_str_2', ax=ax5, line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
ax5.set_xlabel('eta_str')
ax5.set_ylabel('alpha')

ax6 = fig1.add_subplot(grid1[2, 2])
a.plot_continuation('PAR(3)', 'PAR(10)', cont='k_i/eta_str_2', ax=ax6, line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E', ignore=['LP'])
ax6.set_xlabel('eta_str')
ax6.set_ylabel('k_i')

ax7 = fig1.add_subplot(grid1[3, 0])
a.plot_continuation('PAR(3)', 'PAR(9)', cont='k/eta_str_zh', ax=ax7, line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E', ignore=['LP'])
ax7.set_xlabel('eta_str')
ax7.set_ylabel('k')

ax8 = fig1.add_subplot(grid1[3, 1])
a.plot_continuation('PAR(3)', 'PAR(17)', cont='alpha/eta_str_zh', ax=ax8, line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
ax8.set_xlabel('eta_str')
ax8.set_ylabel('alpha')

ax9 = fig1.add_subplot(grid1[3, 2])
a.plot_continuation('PAR(9)', 'PAR(17)', cont='alpha/k_zh', ax=ax9, line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
ax9.set_xlabel('k')
ax9.set_ylabel('alpha')

plt.savefig('results/eta_str_cont.svg', dpi=600, format='svg')

# continuation in k
###################

fig2 = plt.figure(tight_layout=True, figsize=(12, 8))
grid2 = gs.GridSpec(2, 3)

# codim 1
ax0 = fig2.add_subplot(grid2[0, :])
a.plot_continuation('PAR(9)', 'U(3)', cont='k', ax=ax0, line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
a.plot_continuation('PAR(9)', 'U(3)', cont=f'k_lc', ax=ax0, ignore=['BP'], line_color_stable='#148F77')
ax0.set_xlabel('k')
ax0.set_ylabel('r_i')

# codim 2
ax1 = fig2.add_subplot(grid2[1, 0])
a.plot_continuation('PAR(9)', 'PAR(3)', cont='eta_str/k', ax=ax1, line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
ax1.set_xlabel('k')
ax1.set_ylabel('eta_str')

ax2 = fig2.add_subplot(grid2[1, 1])
a.plot_continuation('PAR(9)', 'PAR(17)', cont='alpha/k', ax=ax2, line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
ax2.set_xlabel('k')
ax2.set_ylabel('alpha')

ax3 = fig2.add_subplot(grid2[1, 2])
a.plot_continuation('PAR(9)', 'PAR(18)', cont='delta/k', ax=ax3, line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
ax3.set_xlabel('k')
ax3.set_ylabel('delta')

plt.savefig('results/k_cont.svg', dpi=600, format='svg')

# continuation in delta
#######################

fig3 = plt.figure(tight_layout=True, figsize=(12, 8))
grid3 = gs.GridSpec(2, 3)

# codim 1
ax0 = fig3.add_subplot(grid3[0, :])
a.plot_continuation('PAR(18)', 'U(3)', cont='delta', ax=ax0, line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
a.plot_continuation('PAR(18)', 'U(3)', cont=f'delta_lc', ax=ax0, ignore=['BP'], line_color_stable='#148F77')
ax0.set_xlabel('delta')
ax0.set_ylabel('r_i')

# codim 2
ax1 = fig3.add_subplot(grid3[1, 0])
a.plot_continuation('PAR(18)', 'PAR(3)', cont='eta_str/delta', ax=ax1, line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
ax1.set_xlabel('delta')
ax1.set_ylabel('eta_str')

ax2 = fig3.add_subplot(grid3[1, 1])
a.plot_continuation('PAR(18)', 'PAR(9)', cont='k/delta', ax=ax2, line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
ax2.set_xlabel('delta')
ax2.set_ylabel('k')

ax3 = fig3.add_subplot(grid2[1, 2])
a.plot_continuation('PAR(18)', 'PAR(17)', cont='alpha/delta', ax=ax3, line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
ax3.set_xlabel('delta')
ax3.set_ylabel('alpha')

plt.savefig('results/delta_cont.svg', dpi=600, format='svg')
plt.show()
