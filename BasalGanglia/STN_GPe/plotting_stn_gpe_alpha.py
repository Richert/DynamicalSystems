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

fname = 'results/stn_gpe_alpha.pkl'
a = PyAuto.from_file(fname)

###################################
# plotting of qif_stn_gpe results #
###################################

# continuation in k and eta
###########################

# fig0 = plt.figure(tight_layout=True)
# grid0 = gs.GridSpec(1, 2)
#
# # codim 1 in k
# ax0 = fig0.add_subplot(grid0[0, :])
# a.plot_continuation('PAR(10)', 'U(1)', cont='k', ax=ax0, line_color_stable='#76448A',
#                     line_color_unstable='#5D6D7E', ignore=['UZ'])
# a.plot_continuation('PAR(10)', 'U(3)', cont='k', ax=ax0, line_color_stable='#148F77',
#                     line_color_unstable='#5D6D7E', ignore=['UZ'])
# ax0.set_xlabel('k')
# ax0.set_ylabel('r')
# plt.legend(['STN', 'GPe'])
#
# # codim 1 in eta_i
# ax1 = fig0.add_subplot(grid0[1, :])
# a.plot_continuation('PAR(2)', 'U(1)', cont='eta', ax=ax1, line_color_stable='#76448A',
#                     line_color_unstable='#5D6D7E', ignore=['UZ'])
# a.plot_continuation('PAR(2)', 'U(3)', cont='eta', ax=ax1, line_color_stable='#148F77',
#                     line_color_unstable='#5D6D7E', ignore=['UZ'])
# ax1.set_xlabel('k')
# ax1.set_ylabel('r')
# plt.legend(['STN', 'GPe'])

# continuation in alpha
#######################

fig1 = plt.figure(tight_layout=True)
grid1 = gs.GridSpec(2, 2)

# codim 1
ax0 = fig1.add_subplot(grid1[0, :])
a.plot_continuation('PAR(9)', 'U(1)', cont='alpha', ax=ax0, line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E', ignore=['UZ'])
a.plot_continuation('PAR(9)', 'U(1)', cont='alpha_lc', ax=ax0, line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E', ignore=['UZ'])

ax0.set_xlabel('alpha')
ax0.set_ylabel('r')
ax0.set_title('Excitatory Population')
ax0.set_xlim([0.0, 0.03])

ax1 = fig1.add_subplot(grid1[1, :])
a.plot_continuation('PAR(9)', 'U(3)', cont='alpha', ax=ax1, line_color_stable='#148F77',
                    line_color_unstable='#5D6D7E', ignore=['UZ'])
a.plot_continuation('PAR(9)', 'U(3)', cont='alpha_lc', ax=ax1, line_color_stable='#148F77',
                    line_color_unstable='#5D6D7E', ignore=['UZ'])
ax1.set_xlabel('alpha')
ax1.set_ylabel('r')
ax1.set_title('Inhibitory Population')
ax1.set_xlim([0.0, 0.03])
#plt.legend(['STN', 'GPe'])

# codim 2
# ax1 = fig1.add_subplot(grid1[1, :])
# a.plot_continuation('PAR(10)', 'PAR(9)', cont='k/alpha', ax=ax1, line_color_stable='#76448A',
#                     line_color_unstable='#5D6D7E')
# ax1.set_xlabel('alpha')
# ax1.set_ylabel('k')

plt.show()
plt.savefig('results/stn_gpe_alpha_cont.svg', dpi=600, format='svg')
