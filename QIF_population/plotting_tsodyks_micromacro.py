from scipy.io import loadmat
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np

# preparations
##############
# load matlab variables
data = [{'micro': loadmat(f'data/tsodyks_micro_v{i}.mat'), 'macro': loadmat(f'data/tsodyks_macro_v{i}.mat')}
        for i in range(1, 2)]

# plot settings
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (7, 8)
plt.rcParams['font.size'] = 10.0
plt.rcParams["font.family"] = "Times New Roman"

# spike raster plot indices
N = 10000
cutoff = int(N * 0.05)
n_neurons = 100
idx = np.random.randint(cutoff, N-cutoff, n_neurons)

# plotting
##########

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=6, ncols=3, figure=fig)

# start plotting
for i in range(2):

    micro = data[i]['micro']
    macro = data[i]['macro']

    # plot u*x distribution
    ax1 = fig.add_subplot(grid[0, i])
    ax1.imshow(micro['ud_rec_av'], cmap=plt.get_cmap('RdBu'), aspect='auto', vmin=0.0, vmax=1.0)
    ax1.set_xticks([0, 250, 500, 750, 1000])
    ax1.set_xticklabels(['0', '25', '50', '75', '100'])

    # plot spikes
    ax2 = fig.add_subplot(grid[1, i])
    ax2.eventplot([micro['raster'][0, j][:, 0] for j in idx], colors='k')
    ax2.set_xticklabels(['0', '25', '50', '75', '100'])
    ax2.set_xlim(left=500000, right=1500000)

    # plot firing rates
    ax3 = fig.add_subplot(grid[2, i])
    ax3.plot(micro['times'].squeeze(), micro['r_rec_av'].squeeze(), color='k')
    ax3.plot(micro['times'].squeeze(), macro['r_mac_rec_av'].squeeze(), color='tab:orange')
    ax3.set_xlim(left=0, right=100)

    # plot membrane potential
    ax4 = fig.add_subplot(grid[3, i])
    ax4.plot(micro['times'].squeeze(), micro['v_rec_av'].squeeze(), color='k')
    ax4.plot(micro['times'].squeeze(), macro['v_mac_rec_av'].squeeze(), color='tab:orange')
    ax4.set_xlim(left=0, right=100)

    # plot synaptic depression
    ax5 = fig.add_subplot(grid[4, i])
    ax5.plot(micro['times'].squeeze(), micro['d_rec_av'].squeeze(), color='k')
    ax5.plot(micro['times'].squeeze(), macro['d_mac_rec_av'].squeeze(), color='tab:orange')
    ax5.set_xlim(left=0, right=100)

    # plot synaptic facilitation
    ax6 = fig.add_subplot(grid[5, i])
    ax6.plot(micro['times'].squeeze(), micro['u_rec_av'].squeeze(), color='k')
    ax6.plot(micro['times'].squeeze(), macro['u_mac_rec_av'].squeeze(), color='tab:orange')
    ax6.set_xlim(left=0, right=100)

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

fig.canvas.draw()
fig.savefig('micro_macro.pdf')
plt.show()
