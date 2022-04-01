from matplotlib import gridspec
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_hex
import scipy.io as scio
import numpy as np

# preparations
##############

# load timeseries
path = '/home/rgast/ownCloud/scripts/matlab/gpe/gpe_forced'

# mean-field data
mf_bs = scio.loadmat(f"{path}_mf_bs.mat")
mf_lc = scio.loadmat(f"{path}_mf_lc.mat")

# all-to-all coupled SNN data
ata_small_bs = scio.loadmat(f"{path}_ata_bs_small.mat")
ata_large_bs = scio.loadmat(f"{path}_ata_bs_large.mat")
ata_small_lc = scio.loadmat(f"{path}_ata_lc_small.mat")
ata_large_lc = scio.loadmat(f"{path}_ata_lc_large.mat")

# sparse SNN data
sparse_small_bs = scio.loadmat(f"{path}_sparse_bs_small.mat")
sparse_large_bs = scio.loadmat(f"{path}_sparse_bs_large.mat")
sparse_small_lc = scio.loadmat(f"{path}_sparse_lc_small.mat")
sparse_large_lc = scio.loadmat(f"{path}_sparse_lc_large.mat")

# plot settings
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (6.9, 5.0)
plt.rcParams['font.size'] = 8.0
plt.rcParams['axes.titlesize'] = 8.0
plt.rcParams['axes.labelsize'] = 8.0
plt.rcParams['lines.linewidth'] = 0.7
plt.rcParams['axes.titlepad'] = 1.0
labelpad = 1.0
plt.rcParams['axes.labelpad'] = labelpad
markersize = 15
cmap = sns.color_palette("plasma", as_cmap=False, n_colors=4)

# plotting
##########

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=8, ncols=3, figure=fig)

# group data
bs_data = [ata_small_bs, ata_large_bs, sparse_small_bs, sparse_large_bs]
lc_data = [ata_small_lc, ata_large_lc, sparse_small_lc, sparse_large_lc]
titles = ['N1, p1', 'N2, p1', 'N1, p2', 'N2, p2']
net_sizes = [4000, 40000, 4000, 40000]
rlims = [[0, 78], [0, 106]]
rticks = [[0, 30, 60], [0, 40, 80]]
nticks = [0, 100, 200]

# plot timeseries
k = 0
for mf_data, snn_data, rlim, rtick in zip([mf_bs, mf_lc], [bs_data, lc_data], rlims, rticks):
    for title, snn, N in zip(titles, snn_data, net_sizes):

        # firing rate data
        ax1 = fig.add_subplot(grid[k, 0])
        ax1.plot(snn['rp_rec_av'].squeeze()*1e3, color=to_hex(cmap[0]))
        ax1.plot(mf_data['rp_mf_av'].squeeze()*1e3, color=to_hex(cmap[0]), linestyle='--')
        ax1.plot(snn['ra_rec_av'].squeeze()*1e3, color=to_hex(cmap[-1]))
        ax1.plot(mf_data['ra_mf_av'].squeeze()*1e3, color=to_hex(cmap[-1]), linestyle='--')
        ax1.set_ylim(rlim)
        ax1.set_yticks(rtick)
        ax1.set_xticklabels(["" for _ in range(len(ax1.get_xticks()))])

        # membrane potential data
        ax2 = fig.add_subplot(grid[k, 1])
        ax2.plot(snn['vp_rec_av'].squeeze(), color=to_hex(cmap[0]))
        ax2.plot(mf_data['vp_mf_av'].squeeze(), color=to_hex(cmap[0]), linestyle='--')
        ax2.plot(snn['va_rec_av'].squeeze(), color=to_hex(cmap[-1]))
        ax2.plot(mf_data['va_mf_av'].squeeze(), color=to_hex(cmap[-1]), linestyle='--')
        ax2.set_xticklabels(["" for _ in range(len(ax2.get_xticks()))])

        # plot spikes
        cutoff = int(N * 0.05)
        n_neurons = nticks[-1]
        idx = np.random.randint(cutoff, N - cutoff, n_neurons)
        ax3 = fig.add_subplot(grid[k, 2])
        ax3.eventplot([snn['raster_p'][0, j][:, 0] if len(snn['raster_p'][0, j]) > 0 else np.asarray([])
                       for j in idx], colors='k', lineoffsets=1.0, linelengths=1.0, linewidths=1.0)
        ax3.set_xlim([100000, 300000])
        ax3.set_xticklabels(["" for _ in range(len(ax3.get_xticks()))])
        ax3.set_yticks(nticks)

        # titles
        if k == 0:
            ax1.set_title('A : N1, p1, N1, p2')
            ax2.set_title('Bistable regime')
            ax3.set_title('N2, p1, N2, p2')
        elif k == 4:
            ax1.set_title('B')
            ax2.set_title('Oscillatory regime')

        k += 1

    # labels and ticks
    ax1.set_ylabel('r')
    ax1.set_xlabel('time in ms')
    ax1.set_xticklabels(["", "100", "150", "200", "250", "300"])
    ax2.set_ylabel('v')
    ax2.set_xlabel('time in ms')
    ax2.set_xticklabels(["", "100", "150", "200", "250", "300"])
    ax3.set_ylabel('neuron \#')
    ax3.set_xlabel('time in ms')
    ax3.set_xticklabels(["100", "150", "200", "250", "300"])

# final touches
###############

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving
fig.canvas.draw()
plt.savefig(f'results/gpe_2pop_micro.svg')
plt.show()
