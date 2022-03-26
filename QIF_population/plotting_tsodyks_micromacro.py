from scipy.io import loadmat
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# preparations
##############
# load matlab variables
data = [{'micro': loadmat(f'data/tsodyks_micro_v{i}.mat'), 'macro': loadmat(f'data/tsodyks_macro_v{i}.mat')}
        for i in range(1, 4)]

# plot settings
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 400
plt.rcParams['figure.figsize'] = (7, 8)
plt.rcParams['font.size'] = 8.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 0.7
plt.rcParams["font.family"] = "Times New Roman"

# spike raster plot indices
N = 10000
cutoff = int(N * 0.05)
n_neurons = 100
idx = np.random.randint(cutoff, N-cutoff, n_neurons)

# edges in V
edges = np.linspace(-20, 20, 104)

# plotting
##########

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=6, ncols=3, figure=fig)

# start plotting
for i in range(3):

    micro = data[i]['micro']
    macro = data[i]['macro']

    # plot u*x distribution
    ax1 = fig.add_subplot(grid[0, i])
    im = ax1.imshow(micro['ud_rec_av'], cmap=plt.get_cmap('RdBu'), aspect='auto', vmin=0.0, vmax=1.0, origin='lower')
    ax1.set_xticks([0, 200, 400, 600, 800, 1000])
    ax1.set_xticklabels(['', '', '', '', '', ''])
    ax1.set_yticklabels(['', '0', '0.5', '1.0'])

    # plot spikes
    ax2 = fig.add_subplot(grid[1, i])
    ax2.eventplot([micro['raster'][0, j][:, 0] if len(micro['raster'][0, j]) > 0 else np.asarray([])
                   for j in idx], colors='k', lineoffsets=1.0, linelengths=1.0, linewidths=0.3)
    ax2.set_xlim(left=500000, right=1500000)
    ax2.set_xticks([500000, 700000, 900000, 1100000, 1300000, 1500000])
    ax2.set_xticklabels(['', '', '', '', '', ''])

    # plot firing rates
    ax3 = fig.add_subplot(grid[2, i])
    ax3.plot(micro['times'].squeeze(), micro['r_rec_av'].squeeze(), color='k')
    ax3.plot(micro['times'].squeeze(), macro['r_mac_rec_av'].squeeze(), color='tab:orange')
    ylims = ax3.get_ylim()
    ax3.add_patch(Rectangle((20.0, ylims[0]), width=20.0, height=ylims[1]-ylims[0], alpha=0.2, color='#7f7f7f'))
    ax3.add_patch(Rectangle((60.0, ylims[0]), width=20.0, height=ylims[1]-ylims[0], alpha=0.2, color='#7f7f7f'))
    ax3.set_xlim(left=0, right=100)
    ax3.set_xticks([0, 20, 40, 60, 80, 100])
    ax3.set_xticklabels(['0', '20', '40', '60', '80', '100'])

    # plot membrane potential
    ax4 = fig.add_subplot(grid[3, i])
    # im2 = ax4.imshow(micro['vi_rec_av'], cmap=plt.get_cmap('RdBu'), aspect='auto', vmin=0.0, vmax=1.0, origin='lower')
    # ax4.set_xticks([0, 200, 400, 600, 800, 1000])
    # ax4.set_xticklabels(['', '', '', '', '', ''])
    # ax4.set_yticklabels(['', '-20', '0', '20'])
    ax4.plot(micro['times'].squeeze(), micro['v_rec_av'].squeeze(), color='k')
    ax4.plot(micro['times'].squeeze(), macro['v_mac_rec_av'].squeeze(), color='tab:orange')
    ylims = ax4.get_ylim()
    ax4.add_patch(Rectangle((20.0, ylims[0]), width=20.0, height=ylims[1]-ylims[0], alpha=0.2, color='#7f7f7f'))
    ax4.add_patch(Rectangle((60.0, ylims[0]), width=20.0, height=ylims[1]-ylims[0], alpha=0.2, color='#7f7f7f'))
    ax4.set_xlim(left=0, right=100)
    ax4.set_xticks([0, 20, 40, 60, 80, 100])
    ax4.set_xticklabels(['', '', '', '', '', ''])

    # plot synaptic depression
    ax5 = fig.add_subplot(grid[4, i])
    # v_dist = np.zeros_like(micro['vi_rec_av'])
    # for k in range(v_dist.shape[1]):
    #     center, width = macro['v_mac_rec_av'][0, k], macro['r_mac_rec_av'][0, k] * np.pi
    #     v = center + width * np.tan((np.pi / 2) * (2. * np.arange(1, N + 1) - N - 1) / (N + 1))
    #     v_dist[:, k] = np.histogram(v, edges, density=False)[0]/N
    # im3 = ax5.imshow(v_dist - micro['vi_rec_av'], cmap=plt.get_cmap('RdBu'), aspect='auto', vmin=-0.1, vmax=0.1, origin='lower')
    # ax5.set_xticks([0, 200, 400, 600, 800, 1000])
    # ax5.set_xticklabels(['', '', '', '', '', ''])
    # ax5.set_yticklabels(['', '-20', '0', '20'])
    ax5.plot(micro['times'].squeeze(), micro['d_rec_av'].squeeze(), color='k')
    ax5.plot(micro['times'].squeeze(), macro['d_mac_rec_av'].squeeze(), color='tab:orange')
    ylims = ax5.get_ylim()
    ax5.add_patch(Rectangle((20.0, ylims[0]), width=20.0, height=ylims[1]-ylims[0], alpha=0.2, color='#7f7f7f'))
    ax5.add_patch(Rectangle((60.0, ylims[0]), width=20.0, height=ylims[1]-ylims[0], alpha=0.2, color='#7f7f7f'))
    ax5.set_xlim(left=0, right=100)
    if i == 1:
        leg = ax5.legend([r'$\mathrm{SNN}_{\mathrm{pre}}$ II', r'$\mathrm{FRE}_{\mathrm{Poisson}}$'],
                         loc='center', bbox_to_anchor=(0.5, 0.75), fontsize=plt.rcParams['font.size'])
        leg.set_in_layout(False)
    ax5.set_xticks([0, 20, 40, 60, 80, 100])
    ax5.set_xticklabels(['', '', '', '', '', ''])

    # plot synaptic facilitation
    ax6 = fig.add_subplot(grid[5, i])
    # ax6.plot(micro['times'].squeeze(), micro['u_rec_av'].squeeze() * micro['d_rec_av'].squeeze(), color='k')
    # ax6.plot(micro['times'].squeeze(), macro['u_mac_rec_av'].squeeze() * macro['d_mac_rec_av'].squeeze(),
    #          color='tab:orange')
    # ylims = ax6.get_ylim()
    # ax6.add_patch(Rectangle((20.0, ylims[0]), width=20.0, height=ylims[1] - ylims[0], alpha=0.2, color='#7f7f7f'))
    # ax6.add_patch(Rectangle((60.0, ylims[0]), width=20.0, height=ylims[1] - ylims[0], alpha=0.2, color='#7f7f7f'))
    # ax6.set_xlim(left=0, right=100)
    # ax6.set_xticks([0, 20, 40, 60, 80, 100])
    # ax6.set_xticklabels(['0', '20', '40', '60', '80', '100'])
    ax6.plot(micro['times'].squeeze(), micro['u_rec_av'].squeeze(), color='k')
    ax6.plot(micro['times'].squeeze(), macro['u_mac_rec_av'].squeeze(), color='tab:orange')
    if i == 0:
        ax6.set_yticks([0.95, 1.0, 1.05])
    elif i == 1:
        ax6.set_yticks([0.89, 0.9, 0.91])
    ylims = ax6.get_ylim()
    ax6.add_patch(Rectangle((20.0, ylims[0]), width=20.0, height=ylims[1]-ylims[0], alpha=0.2, color='#7f7f7f'))
    ax6.add_patch(Rectangle((60.0, ylims[0]), width=20.0, height=ylims[1]-ylims[0], alpha=0.2, color='#7f7f7f'))
    ax6.set_xlim(left=0, right=100)
    ax6.set_xticks([0, 20, 40, 60, 80, 100])
    ax6.set_xticklabels(['0', '20', '40', '60', '80', '100'])

    # axis labels etc.
    if i == 0:
        ax1.set_ylabel(r'$U_i X_i$')
        ax2.set_ylabel('neuron \#')
        ax3.set_ylabel(r'$r$')
        ax4.set_ylabel(r'$v$')
        ax5.set_ylabel(r'$x$')
        ax6.set_ylabel(r'$u$')
        ax1.set_title(r'\textbf{A:} Short-term depression')
    elif i == 1:
        ax6.set_xlabel('time')
        ax1.set_title(r'\textbf{B:} Short-term facilitation')
    elif i == 2:
        ax1.set_title(r'\textbf{C:} Short-term depr. + facil.')

# colorbar
fig.colorbar(im, ax=ax1, shrink=0.8)

# padding
fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.03, hspace=0., wspace=0.)

fig.canvas.draw()
fig.savefig('micro_macro.pdf')
plt.show()
