from matplotlib import gridspec
import matplotlib.pyplot as plt
import pickle
from pyrates.utility.pyauto import PyAuto
import sys
sys.path.append('../')

# preparations
##############

fname = 'stn_gpe_3pop'

# load simulation data
data = pickle.load(open(f"results/{fname}_sims.p", "rb"))

# load pyauto data
path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"
a = PyAuto.from_file(f"results/{fname}_conts.pkl", auto_dir=auto_dir)

# plot settings
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (5.25, 4.0)
plt.rcParams['font.size'] = 8.0
plt.rcParams['axes.titlesize'] = 8.0
plt.rcParams['axes.labelsize'] = 8.0
plt.rcParams['lines.linewidth'] = 0.7
plt.rcParams['axes.titlepad'] = 0.0
labelpad = 1.0
plt.rcParams['axes.labelpad'] = labelpad
markersize = 25

############
# plotting #
############

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=5, ncols=12, figure=fig)

# 2d: k_pe x k_gp
ax = fig.add_subplot(grid[:3, :5])
ax = a.plot_continuation('PAR(22)', 'PAR(5)', cont=f'k_gp/k_pe:lp1', ax=ax, line_color_stable='#3689c9',
                         line_color_unstable='#3689c9', default_size=markersize,
                         line_style_unstable='solid', ignore=['UZ', 'GH', 'ZH'])
ax = a.plot_continuation('PAR(22)', 'PAR(5)', cont=f'k_gp/k_pe:hb1', ax=ax, line_color_stable='#148F77',
                         line_color_unstable='#148F77', default_size=markersize,
                         line_style_unstable='solid', ignore=['UZ', 'GH', 'ZH'])
ax = a.plot_continuation('PAR(22)', 'PAR(5)', cont=f'k_gp/k_pe:hb2', ax=ax, line_color_stable='#ee2b2b',
                         line_color_unstable='#ee2b2b', default_size=markersize,
                         line_style_unstable='solid', ignore=['UZ', 'GH', 'ZH'])
# ax = a.plot_continuation('PAR(22)', 'PAR(5)', cont=f'k_gp/k_pe:hb3', ax=ax, line_color_stable='#ee2b2b',
#                          line_color_unstable='#ee2b2b', default_size=markersize,
#                          line_style_unstable='solid', ignore=['UZ'])
ax.set_ylabel(r'$k_{pe}$', labelpad=labelpad)
ax.set_xlabel(r'$k_{gp}$', labelpad=labelpad)
ax.set_title('2D bifurcation diagram')
ax.set_xlim([0.0, 20.0])
ax.set_ylim([0.0, 25.0])

# 1D continuation in k_pe for k_gp = 5.0
ax = fig.add_subplot(grid[0, 5:])
ax = a.plot_continuation('PAR(5)', 'U(3)', cont=f'k_pe:1', ax=ax, default_size=markersize, ignore=['UZ', 'GH'])
ax = a.plot_continuation('PAR(5)', 'U(3)', cont=f'k_pe:1:lc1', ax=ax, default_size=markersize, ignore=['UZ'],
                         line_color_stable='#148F77', line_color_unstable='#148F77')
ax = a.plot_continuation('PAR(5)', 'U(3)', cont=f'k_pe:1:lc2', ax=ax, default_size=markersize, ignore=['UZ'],
                         line_color_stable='#148F77', line_color_unstable='#148F77')
# ax = a.plot_continuation('PAR(5)', 'U(3)', cont=f'k_pe:1:lc3', ax=ax, default_size=markersize, ignore=['UZ'],
#                          line_color_stable='#148F77', line_color_unstable='#148F77')
ax.set_xlabel('$k_{pe}$', labelpad=labelpad)
ax.set_ylabel('', labelpad=labelpad)
ax.set_title('1D bifurcation diagrams')
# ax.set_xlim([0.0, 5.0])
# ax.set_ylim([0.0, 0.15])
# ax.set_yticks([0, 0.06, 0.12])
# ax.set_yticklabels(["0", "60", "120"])

# 1D continuation in k_gp for k_pe = 8.0
ax = fig.add_subplot(grid[1, 5:])
ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'k_gp:1', ax=ax, default_size=markersize, ignore=['UZ'])
ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'k_gp:1:lc1', ax=ax, default_size=markersize, ignore=['UZ'],
                         line_color_stable='#148F77', line_color_unstable='#148F77')
# ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'k_gp:1:lc2', ax=ax, default_size=markersize, ignore=['UZ'],
#                          line_color_stable='#148F77', line_color_unstable='#148F77')
ax.set_xlabel('$k_{gp}$', labelpad=labelpad)
ax.set_ylabel('r', labelpad=labelpad)
# ax.set_xlim([1.0, 13.0])
# ax.set_ylim([0.0, 0.2])
# ax.set_yticks([0, 0.1, 0.2])
# ax.set_yticklabels(["0", "100", "200"])

# 1D continuation in k_gp for k_pe = 16.0
ax = fig.add_subplot(grid[2, 5:])
ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'k_gp:2', ax=ax, default_size=markersize, ignore=['UZ'])
ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'k_gp:2:lc1', ax=ax, default_size=markersize, ignore=['UZ'],
                         line_color_stable='#148F77', line_color_unstable='#148F77')
ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'k_gp:2:lc2', ax=ax, default_size=markersize, ignore=['UZ'],
                         line_color_stable='#148F77', line_color_unstable='#148F77')
# ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'k_gp:2:lc3', ax=ax, default_size=markersize, ignore=['UZ'],
#                          line_color_stable='#148F77', line_color_unstable='#148F77')
ax.set_xlabel('$k_{gp}$', labelpad=labelpad)
ax.set_ylabel('', labelpad=labelpad)
# ax.set_xlim([13.5, 15.5])
# ax.set_ylim([0.0, 0.1])
# ax.set_yticks([0, 0.05, 0.1])
# ax.set_yticklabels(["0", "50", "100"])

# plt time series and psd profiles
##################################

# rates = data['results']
# map = data['map']
# psds = data['psds']
# ax_indices = [(3, 0), (3, 6), (4, 0), (4, 6)]
#
# for i, key in enumerate(map.index):
#
#     # extract data
#     k_pe = map.loc[key, 'k_pe']
#     k_gp = map.loc[key, 'k_gp']
#     row = ax_indices[i][0]
#     col = ax_indices[i][1]
#
#     # plot timeseries
#     ax1 = fig.add_subplot(grid[row, col:col+4])
#     ax1.plot(rates.index[100000:105000], rates.loc[10.0:10.49999, ('r_e', key)])
#     ax1.plot(rates.index[100000:105000], rates.loc[10.0:10.49999, ('r_i', key)])
#     #ax1.plot(rates.index[100000:105000], rates.loc[10.0:10.49999, ('r_a', key)])
#     ax1.set_title('r')
#     #ax1.set_ylim([18.0, 180.0])
#
#     # plot psd
#     ax2 = fig.add_subplot(grid[row, col+4:col+6])
#     ax2.plot(psds['freq_stn'][i], psds['pow_stn'][i])
#     ax2.plot(psds['freq_gpe'][i], psds['pow_gpe'][i])
#     ax2.set_title('PSD')
#     ax2.set_in_layout(False)
#     ax2.set_xlim([0.0, 120.0])
#     ax2.set_xticks([0, 50, 100])
#     #ax2.set_ylim([-5.0, 275] if i == 2 else [-5.0, 50])
#
# ax1.set_xlabel('time in ms')
# ax2.set_xlabel('f in Hz')

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

fig.canvas.draw()
plt.savefig(f'results/{fname}_bf.svg')
plt.show()
