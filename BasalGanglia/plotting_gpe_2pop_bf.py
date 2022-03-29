from matplotlib import gridspec
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_hex
import pickle
from pyauto import PyAuto
import sys
sys.path.append('../')

# preparations
##############

fname = 'gpe_2pop'

# load simulation data
lc_data = pickle.load(open(f"results/{fname}_stim1.p", "rb"))['results']
bs_data = pickle.load(open(f"results/{fname}_stim2.p", "rb"))['results']

# load pyauto data
path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"
a = PyAuto.from_file(f"results/{fname}_conts.pkl", auto_dir=auto_dir)

# plot settings
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (4.5, 5.0)
plt.rcParams['font.size'] = 8.0
plt.rcParams['axes.titlesize'] = 8.0
plt.rcParams['axes.labelsize'] = 8.0
plt.rcParams['lines.linewidth'] = 0.7
plt.rcParams['axes.titlepad'] = 1.0
labelpad = 1.0
plt.rcParams['axes.labelpad'] = labelpad
markersize = 15
cmap = sns.color_palette("plasma", as_cmap=False, n_colors=4)

############
# plotting #
############

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=5, ncols=5, figure=fig)

# A: 2D bifurcation diagrams
############################

# 2d: k_pp x eta_p
ax1 = fig.add_subplot(grid[1, 3:])
ax1 = a.plot_continuation('PAR(2)', 'PAR(6)', cont='k_pp/eta_p:hb1', ax=ax1, line_style_unstable='solid',
                          default_size=markersize)
ax1.set_xlabel(r'$\bar \eta_p$', labelpad=labelpad)
ax1.set_ylabel(r'$J_{pp}$', labelpad=labelpad)
ax1.set_xlim([0.0, 60.0])
ax1.set_ylim([0.0, 8.0])
ax1.set_yticks([0, 3, 6])
ax1.set_yticklabels(["0", "30", "60"])
ax1.set_title('D')

# 2d: k_pa x eta_p
ax2 = fig.add_subplot(grid[3, 3:])
ax2 = a.plot_continuation('PAR(2)', 'PAR(8)', cont='k_pa/eta_p:lp1', ax=ax2, line_style_unstable='solid',
                          line_color_stable='#3689c9', line_color_unstable='#3689c9', default_size=markersize)
ax2 = a.plot_continuation('PAR(2)', 'PAR(8)', cont='k_pa/eta_p:hb1', ax=ax2, line_style_unstable='solid',
                          default_size=markersize)
ax2.set_xlabel(r'$\eta_p$', labelpad=labelpad)
ax2.set_ylabel(r'$J_{pa}$', labelpad=labelpad)
ax2.set_xlim([0.0, 40.0])
ax2.set_ylim([0.0, 8.0])
ax2.set_yticks([0, 3, 6])
ax2.set_yticklabels(["0", "30", "60"])
ax2.set_title('G')

# B: 1D bifurcation diagrams
############################

# 1D: eta_p for default connectivity
ax0 = fig.add_subplot(grid[0, :3])
ax0 = a.plot_continuation('PAR(2)', 'U(2)', cont='eta_p:0', ax=ax0, line_color_stable=to_hex(cmap[0]),
                          line_color_unstable=to_hex(cmap[0]), default_size=markersize)
ax0 = a.plot_continuation('PAR(2)', 'U(4)', cont='eta_p:0', ax=ax0, line_color_stable=to_hex(cmap[-1]),
                          line_color_unstable=to_hex(cmap[-1]), default_size=markersize)
ax0.set_xlabel(r'$\eta_p$', labelpad=labelpad)
ax0.set_ylabel(r'$r$', labelpad=labelpad)
ax0.set_xlim([-50.0, 50.0])
ax0.set_ylim([0.0, 0.12])
ax0.set_yticks([0, 0.05, 0.1])
ax0.set_yticklabels(["0", "50", "100"])
ax0.set_title('A Default connectivity')

# 1D: eta_p for k_pp = 5.0
ax3 = fig.add_subplot(grid[1, :3])
ax3 = a.plot_continuation('PAR(2)', 'U(2)', cont='eta_p:2', ax=ax3, line_color_stable=to_hex(cmap[0]),
                          line_color_unstable=to_hex(cmap[0]), default_size=markersize)
ax3 = a.plot_continuation('PAR(2)', 'U(4)', cont='eta_p:2', ax=ax3, line_color_stable=to_hex(cmap[-1]),
                          line_color_unstable=to_hex(cmap[-1]), default_size=markersize)
ax3 = a.plot_continuation('PAR(2)', 'U(2)', cont='eta_p:2:lc', ax=ax3, line_color_stable=to_hex(cmap[1]),
                          line_color_unstable=to_hex(cmap[1]), default_size=markersize)
ax3 = a.plot_continuation('PAR(2)', 'U(4)', cont='eta_p:2:lc', ax=ax3, line_color_stable=to_hex(cmap[-2]),
                          line_color_unstable=to_hex(cmap[-2]), default_size=markersize)
ax3.set_xlabel(r'$\eta_p$', labelpad=labelpad)
ax3.set_ylabel(r'$r$', labelpad=labelpad)
ax3.set_xlim([0.0, 45.0])
ax3.set_ylim([0.0, 0.13])
ax3.set_yticks([0, 0.05, 0.1])
ax3.set_yticklabels(["0", "50", "100"])
ax3.set_title('C Increased GPe-p self-inhibition')

# 1D: eta_p for k_pa = 4.0
ax4 = fig.add_subplot(grid[3, :3])
ax4 = a.plot_continuation('PAR(2)', 'U(2)', cont='eta_p:3', ax=ax4, line_color_stable=to_hex(cmap[0]),
                          line_color_unstable=to_hex(cmap[0]), default_size=markersize)
ax4 = a.plot_continuation('PAR(2)', 'U(4)', cont='eta_p:3', ax=ax4, line_color_stable=to_hex(cmap[-1]),
                          line_color_unstable=to_hex(cmap[-1]), default_size=markersize, ignore=['HB'])
ax4 = a.plot_continuation('PAR(2)', 'U(2)', cont='eta_p:3:lc', ax=ax4, line_color_stable=to_hex(cmap[1]),
                          line_color_unstable=to_hex(cmap[1]), ignore=['LP', 'TR'], default_size=markersize)
# ax4 = a.plot_continuation('PAR(2)', 'U(4)', cont='eta_p:3:lc', ax=ax4, line_color_stable=to_hex(cmap[-2]),
#                           line_color_unstable=to_hex(cmap[-2]))
ax4.set_xlabel(r'$\eta_p$', labelpad=labelpad)
ax4.set_ylabel(r'$r$', labelpad=labelpad)
ax4.set_xlim([10.0, 30.0])
ax4.set_ylim([0.0, 0.085])
ax4.set_yticks([0, 0.03, 0.06])
ax4.set_yticklabels(["0", "30", "60"])
ax4.set_title('F Increased inhibition of GPe-p by GPe-a')

# C: time series
################

idx_l, idx_u = (20000, 30000)
t0, t1 = (2000.0, 2999.99)

# firing rate plot for k_pp = 5.0 and eta_p = 30.0 + 10.0
ax5 = fig.add_subplot(grid[2, :])
ax5.plot(lc_data.index[idx_l:idx_u], lc_data.loc[t0:t1, 'r_i'], color=to_hex(cmap[0]))
ax5.plot(lc_data.index[idx_l:idx_u], lc_data.loc[t0:t1, 'r_a'], color=to_hex(cmap[-1]))
ax5.set_xlabel('time in ms')
ax5.set_ylabel('r')
ax5.set_ylim([0, 0.1])
ax5.set_yticks([0, 0.03, 0.06, 0.09])
ax5.set_yticklabels(["0", "30", "60", "90"])
ax5.set_title('E')

# firing rate plot for k_pa = 4.86 and eta_p = 20.0 +/- 5.0
ax6 = fig.add_subplot(grid[4, :])
ax6.plot(bs_data.index[idx_l:idx_u], bs_data.loc[t0:t1, 'r_i'], color=to_hex(cmap[0]))
ax6.plot(bs_data.index[idx_l:idx_u], bs_data.loc[t0:t1, 'r_a'], color=to_hex(cmap[-1]))
ax6.set_xlabel('time in ms')
ax6.set_ylabel('r')
ax6.set_ylim([0, 0.1])
ax6.set_yticks([0, 0.03, 0.06, 0.09])
ax6.set_yticklabels(["0", "30", "60", "90"])
ax6.set_title('H')

# final touches
###############

# first axis
ax7 = fig.add_subplot(grid[0, 3:])
ax7.set_title(r'$\mathrm{ms}^2$')
ax7.set_xlabel('ms')
ax7.set_ylabel(r'$J_{pp}$, $J_{pa}$')

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving
fig.canvas.draw()
plt.savefig(f'results/{fname}_bf.svg')

plt.show()
