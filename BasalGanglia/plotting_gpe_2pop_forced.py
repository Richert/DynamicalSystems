from matplotlib import gridspec
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_hex
import pickle
import scipy.io as scio
from pyrates.utility.pyauto import PyAuto
import sys
sys.path.append('../')

# preparations
##############

# load PAC matrices
pac_path = '/home/rgast/ownCloud/scripts/matlab/gpe/PAC_PPC_plotting_data'
pac_ss = scio.loadmat(f"{pac_path}_ss.mat")
pac_lc = scio.loadmat(f"{pac_path}_lc.mat")
pac_bs = scio.loadmat(f"{pac_path}_bs.mat")

# load time series
ts_path = f"results/gpe_2pop_pac"
ss_data = pickle.load(open(f"{ts_path}_ss.p", "rb"))
lc_data = pickle.load(open(f"{ts_path}_lc.p", "rb"))
bs_data = pickle.load(open(f"{ts_path}_bs.p", "rb"))

# load pyauto data
path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"
fname = 'results/gpe_2pop_forced_lc.pkl'
a = PyAuto.from_file(fname, auto_dir=auto_dir)

# plot settings
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (6.9, 7.0)
plt.rcParams['font.size'] = 8.0
plt.rcParams['axes.titlesize'] = 8.0
plt.rcParams['axes.labelsize'] = 8.0
plt.rcParams['lines.linewidth'] = 0.7
plt.rcParams['axes.titlepad'] = 1.0
labelpad = 1.0
plt.rcParams['axes.labelpad'] = labelpad
markersize = 15
#cmap = sns.diverging_palette(145, 300, s=60, as_cmap=False, n=2)
cmap = sns.color_palette("plasma", as_cmap=False, n_colors=4)
cmap2 = sns.color_palette("plasma", as_cmap=True, n_colors=50)

############
# plotting #
############

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=15, ncols=3, figure=fig)

# A: PAC and PPC matrices
#########################

titles = [('mean PAC', 'PAC-PPC correlation'), ('A', 'B'), ('C', 'D')]
pac_data = [pac_ss, pac_bs, pac_lc]
cbar_shrink = 0.7
cbar_pad = -0.02

for i, (title, data) in enumerate(zip(titles, pac_data)):

    # extract pac-related data
    alpha = data['stim_amps_unique']
    omega = data['stim_freqs_unique']
    pac = data['PAC_mean']
    pac_ppc = data['PAC_PPC_corr']

    ax1 = fig.add_subplot(grid[i*4:(i+1)*4, 0])
    im1 = ax1.imshow(pac, cmap=cmap2, aspect='auto', origin='lower')
    ax1.set_xticks([0, 10, 20, 30])
    ax1.set_yticks([0, 10, 20])
    ax1.set_xticklabels(['50', '60', '70', '80'])
    ax1.set_yticklabels(['0', '1', '2'])
    ax1.set_ylabel(r'$\alpha$')
    fig.colorbar(im1, ax=ax1, shrink=cbar_shrink, pad=cbar_pad)
    ax1.set_title(title[0])

    ax2 = fig.add_subplot(grid[i*4:(i+1)*4, 1])
    im2 = ax2.imshow(pac_ppc, cmap=cmap2, aspect='auto', origin='lower', vmin=-0.7, vmax=0.7)
    ax2.set_xticks([0, 10, 20, 30])
    ax2.set_yticks([0, 10, 20])
    ax2.set_xticklabels(['50', '60', '70', '80'])
    ax2.set_yticklabels(['0', '1', '2'])
    #ax2.set_ylabel(r'$\alpha$')
    fig.colorbar(im2, ax=ax2, shrink=cbar_shrink, pad=cbar_pad)
    ax2.set_title(title[1])

ax1.set_xlabel(r'$\omega$')
ax2.set_xlabel(r'$\omega$')

# B: Time series and PSDs
#########################

ts_data = [ss_data, bs_data, lc_data]
idx_l, idx_u = (82000, 88000)
t0, t1 = (8200.0, 8799.99)
styles = ['solid', 'dotted', 'dashed']

for i, data in enumerate(ts_data):
    for j in range(len(data['rates'])):

        # extract time series and PSDs
        alpha = data['params'][j]['alpha']
        omega = data['params'][j]['omega']
        proto = data['rates'][j]['proto']*1e3
        arky = data['rates'][j]['arky']*1e3
        f = data['psds'][j]['freq']
        p = data['psds'][j]['pow'][0]

        # plot time series
        ax1 = fig.add_subplot(grid[(i*4)+j, 2])
        ax1.plot(proto.index[idx_l:idx_u], proto.loc[t0:t1, :], color=to_hex(cmap[0]))
        ax1.plot(arky.index[idx_l:idx_u], arky.loc[t0:t1, :], color=to_hex(cmap[-1]))
        ax1.set_ylabel('r')
        ax1.set_xticklabels(["", "", ""])
        if i+j == 0:
            ax1.set_title('Firing rate dynamics')

        # plot PSD
        # ax2 = fig.add_subplot(grid[(i * 4) + 3, 2])
        # ax2.plot(f, p, linestyle=styles[j])

    # adjust axis labels
    ax1.set_xticklabels(["", "200", "400", "600", "800"])
    ax1.set_xlabel('time in ms')
    # ax2.set_xlabel('f in Hz')
    # ax2.set_ylabel('PSD')

# C: Bifurcation diagram
########################

ax = fig.add_subplot(grid[12:, :2])

# continuation of the torus bifurcation in alpha and omega
i = 1
while i < 11:
    try:
        ax = a.plot_continuation('PAR(11)', 'PAR(21)', cont=f'c1:omega/alpha/TR{i}', ax=ax, ignore=['UZ', 'BP'],
                                 line_color_stable='#148F77', line_color_unstable='#148F77',
                                 custom_bf_styles={'R1': {'marker': 's', 'color': 'k'},
                                                   'R2': {'marker': 'o', 'color': 'k'},
                                                   'R3': {'marker': 'v', 'color': 'k'},
                                                   'R4': {'marker': 'd', 'color': 'k'}},
                                 line_style_unstable='solid', default_size=markersize)
        i += 1
    except KeyError:
        i += 1
i = 1
while i < 11:
    try:
        ax = a.plot_continuation('PAR(11)', 'PAR(21)', cont=f'c1:omega/alpha/PD{i}', ax=ax, ignore=['UZ', 'BP'],
                                 line_color_stable='#3689c9', line_color_unstable='#3689c9',
                                 line_style_unstable='solid', default_size=markersize)
        i += 1
    except KeyError:
        i += 1

ax.set_ylabel(r'$\alpha$')
ax.set_xlabel(r'$\omega$')
ax.set_title('2D bifurcation diagram')
#ax.set_yticklabels([label._y + 5.0 for label in ax.get_yticklabels()])
ax.set_ylim([0.0, 2.0])
ax.set_xlim([55.0, 85.0])

# final touches
###############

# dummy axis with bifurcation labels
ax = fig.add_subplot(grid[12:, 2:])
ax.set_title('Torus, Period doubling')
ax.set_xlabel('Fold of LC')
ax.set_ylabel('R1, R2, R3, R4')

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving
fig.canvas.draw()
plt.savefig(f'results/gpe_2pop_pac.svg')
plt.show()
