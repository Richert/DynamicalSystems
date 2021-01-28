from matplotlib import gridspec
import matplotlib.pyplot as plt
from pyrates.utility.pyauto import PyAuto
import sys
import pickle
sys.path.append('../')

# preparations
##############

# auto data loading
path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"
a_base = PyAuto.from_file('results/stn_gpe_3pop_conts.pkl', auto_dir=auto_dir)

# time series and psd profiles
data = pickle.load(open("results/stn_gpe_3pop_reductions_sims.p", "rb"))

# plot settings
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (5.25, 5.0)
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

# reduction conditions
fnames = ['stn_gpe_2pop', 'stn_gpe_3pop_fre', 'stn_gpe_3pop_noax', 'stn_gpe_3pop_nosyns']
grid_idx = [(0, 0), (0, 6), (3, 0), (3, 6)]
titles = ['A: no GPe-a feedback', 'B: FRE reduction', 'C: no axonal delays', 'D: instantaneous synapses']
titles2 = ['a', 'b', 'c', 'd']
grid_idx2 = [(6, 0), (6, 6), (7, 0), (7, 6)]

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=8, ncols=12, figure=fig)

# loop over different model reductions
######################################

for i, (f, idx, title, idx2) in enumerate(zip(fnames, grid_idx, titles, grid_idx2)):

    # load pyauto data
    a = PyAuto.from_file(f"results/{f}_conts.pkl", auto_dir=auto_dir)

    # plot 2d bifurcation diagram: k_pe x k_gp
    ax = fig.add_subplot(grid[idx[0]:idx[0]+3, idx[1]:idx[1]+6])
    ax = a.plot_continuation('PAR(22)', 'PAR(5)', cont=f'k_gp/k_pe:hb1', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize,
                             line_style_unstable='solid', ignore=['UZ', 'GH', 'ZH'])
    ax = a_base.plot_continuation('PAR(22)', 'PAR(5)', cont=f'k_gp/k_pe:hb1', ax=ax, line_color_stable='#148F77',
                                  line_color_unstable='#148F77', default_size=markersize,
                                  line_style_stable='dotted', ignore=['UZ', 'GH', 'ZH'])
    try:
        ax = a.plot_continuation('PAR(22)', 'PAR(5)', cont=f'k_gp/k_pe:lp1', ax=ax, line_color_stable='#3689c9',
                                 line_color_unstable='#3689c9', default_size=markersize,
                                 line_style_unstable='solid', ignore=['UZ', 'GH', 'ZH'])
        ax = a_base.plot_continuation('PAR(22)', 'PAR(5)', cont=f'k_gp/k_pe:lp1', ax=ax, line_color_stable='#3689c9',
                                      line_color_unstable='#3689c9', default_size=markersize,
                                      line_style_stable='dotted', ignore=['UZ', 'GH', 'ZH'])
    except KeyError:
        pass
    try:
        ax = a.plot_continuation('PAR(22)', 'PAR(5)', cont=f'k_gp/k_pe:lp2', ax=ax, line_color_stable='#3689c9',
                                 line_color_unstable='#3689c9', default_size=markersize,
                                 line_style_unstable='solid', ignore=['UZ', 'GH', 'ZH'])
        ax = a_base.plot_continuation('PAR(22)', 'PAR(5)', cont=f'k_gp/k_pe:lp2', ax=ax, line_color_stable='#3689c9',
                                      line_color_unstable='#3689c9', default_size=markersize,
                                      line_style_stable='dotted', ignore=['UZ', 'GH', 'ZH'])
    except KeyError:
        pass
    try:
        ax = a.plot_continuation('PAR(22)', 'PAR(5)', cont=f'k_gp/k_pe:hb2', ax=ax, line_color_stable='#ee2b2b',
                                 line_color_unstable='#ee2b2b', default_size=markersize,
                                 line_style_unstable='solid', ignore=['UZ', 'GH', 'ZH'])
        ax = a_base.plot_continuation('PAR(22)', 'PAR(5)', cont=f'k_gp/k_pe:hb2', ax=ax, line_color_stable='#ee2b2b',
                                      line_color_unstable='#ee2b2b', default_size=markersize,
                                      line_style_stable='dotted', ignore=['UZ', 'GH', 'ZH'])
    except KeyError:
        pass
    try:
        ax = a.plot_continuation('PAR(22)', 'PAR(5)', cont=f'k_gp/k_pe:hb3', ax=ax, line_color_stable='#148F77',
                                 line_color_unstable='#148F77', default_size=markersize,
                                 line_style_unstable='solid', ignore=['UZ', 'GH', 'ZH'])
        ax = a_base.plot_continuation('PAR(22)', 'PAR(5)', cont=f'k_gp/k_pe:hb3', ax=ax, line_color_stable='#148F77',
                                      line_color_unstable='#148F77', default_size=markersize,
                                      line_style_stable='dotted', ignore=['UZ', 'GH', 'ZH'])
    except KeyError:
        pass
    ax.set_ylabel(r'$k_{pe}$', labelpad=labelpad)
    ax.set_xlabel(r'$k_{gp}$', labelpad=labelpad)
    ax.set_xlim([0.0, 20.0])
    ax.set_ylim([0.0, 25.0])
    ax.set_title(title)

    # plot firing rate timeseries
    rates = data['results'][i]
    row = idx2[0]
    col = idx2[1]
    ax2 = fig.add_subplot(grid[row, col:col + 4])
    ax2.plot(rates.index[100000:105000], rates.loc[10.0:10.49999, 'r_e'])
    ax2.plot(rates.index[100000:105000], rates.loc[10.0:10.49999, 'r_i'])
    # ax2.plot(rates.index[10000:15000], rates.loc[1.0:1.49999, 'r_e'])
    # ax2.plot(rates.index[10000:15000], rates.loc[1.0:1.49999, 'r_i'])
    # ax1.plot(rates.index[100000:105000], rates.loc[10.0:10.49999, ('r_a', key)])
    ax2.set_ylabel(titles2[i])
    ax2.set_title('r')

    # plot psd
    psds = data['psds']
    ax3 = fig.add_subplot(grid[row, col + 4:col + 6])
    ax3.plot(psds['freq_stn'][i], psds['pow_stn'][i])
    ax3.plot(psds['freq_gpe'][i], psds['pow_gpe'][i])
    ax3.set_title('PSD')
    ax3.set_in_layout(False)
    ax3.set_xlim([0.0, 120.0])
    ax3.set_xticks([0, 50, 100])

ax2.set_xlabel('time in ms')
ax3.set_xlabel('f in Hz')
ax3.set_yticklabels(['0.0', '0', '5e-4'])

# padding
fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.03, hspace=0., wspace=0.)

fig.canvas.draw()
plt.savefig(f'results/stn_gpe_3pop_reductions_bf.svg')
plt.show()
