from scipy.io import loadmat
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
from pyrates.utility.pyauto import PyAuto
import sys
sys.path.append('../')

# preparations
##############

# load matlab variables
data = [loadmat(f'data/tsodyks_allscales_v{i}.mat') for i in range(1, 5)]

# load python data
a = PyAuto.from_file('results/tsodyks_poisson.pkl', auto_dir='~/PycharmProjects/auto-07p')

# plot settings
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.figsize'] = (7, 7)
plt.rcParams['font.size'] = 10.0
plt.rcParams["font.family"] = "Times New Roman"
markersize = 40

# spike raster plot indices
N = 10000
cutoff = int(N * 0.05)
n_neurons = 50
idx = np.random.randint(cutoff, N-cutoff, n_neurons)

# continuation specific variables
etas = ['eta_3', 'eta_4', 'eta_1', 'eta_2']
xlims = [[-1.0, 0.2], [-1.0, 0.2], [-1.0, -0.7], [-1.0, 2.0]]
ylims = [[0.0, 0.8], [0.0, 0.8], [0.0, 0.5], [0.0, 2.0]]

# plotting
##########

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=8, ncols=3, figure=fig)

# start plotting
for i in range(4):

    # plot 1D continuation in eta
    ax1 = fig.add_subplot(grid[2*i:2*(i+1), 0])
    ax1 = a.plot_continuation('PAR(1)', 'U(1)', cont=etas[i], ax=ax1, default_size=markersize)
    if i > 1:
        ax1 = a.plot_continuation('PAR(1)', 'U(1)', cont=f"{etas[i]}:lc", ax=ax1, default_size=markersize,
                                  ignore=['BP'], line_color_stable='#7f7f7f', line_color_unstable='#7f7f7f')
    ax1.set_ylabel(r'$r$')
    ax1.set_xlabel('')
    ax1.set_xlim(xlims[i])
    ax1.set_ylim(ylims[i])

    # plot firing rates
    ax2 = fig.add_subplot(grid[2*i, 1:])
    ax2.plot(data[i]['times'].squeeze(), data[i]['r_rec_av'].squeeze(), color='k')
    ax2.plot(data[i]['times'].squeeze(), data[i]['r_mes_rec_av'].squeeze(), color='tab:purple')
    ax2.plot(data[i]['times'].squeeze(), data[i]['r_poisson_rec_av'].squeeze(), color='tab:orange')
    ax2.set_ylabel(r'$r$')
    ax2.set_xlim([0, 280])

    # plot spikes
    ax3 = fig.add_subplot(grid[2*i+1, 1:])
    ax3.eventplot([data[i]['raster'][0, j][:, 0] if len(data[i]['raster'][0, j]) > 0 else np.asarray([])
                   for j in idx], colors='k')
    ax3.set_xlim([500000, 3300000])
    ax3.set_ylabel('neuron \#')
    ax3.set_xticklabels(['0', '50', '100', '150', '200', '250'])

ax1.set_xlabel(r'$\bar \eta$')
ax3.set_xlabel('time')

fig.canvas.draw()
fig.savefig('meanfield_bf.pdf')
plt.show()
