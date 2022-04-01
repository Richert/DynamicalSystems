import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from matplotlib.colors import to_hex
import scipy.io as scio

# preparations
##############

# plot settings
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (4.5, 3.0)
plt.rcParams['font.size'] = 8.0
plt.rcParams['axes.titlesize'] = 8.0
plt.rcParams['axes.labelsize'] = 8.0
plt.rcParams['lines.linewidth'] = 0.6
plt.rcParams['axes.titlepad'] = 1.0
labelpad = 1.0
plt.rcParams['axes.labelpad'] = labelpad
markersize = 15
cmap = sns.color_palette("plasma", as_cmap=False, n_colors=4)

# load data
path = '/home/rgast/MatlabProjects/QIFSimulations/matlab_scripts/neco'
data = scio.loadmat(f"{path}/neco_fig1_data.mat")
N = data['raster'].shape[1]

# plotting
##########

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=5, ncols=7, figure=fig)

# plot spikes
cutoff = int(N * 0.05)
n_neurons = 50
idx = np.random.randint(cutoff, N - cutoff, n_neurons)
ax = fig.add_subplot(grid[0, :3])
ax.eventplot([data['raster'][0, j][:, 0] if len(data['raster'][0, j]) > 0 else np.asarray([])
               for j in idx], colors='k', lineoffsets=1.0, linelengths=1.0, linewidths=0.7)
# ax.set_xlim([100000, 300000])
# ax.set_xticklabels(["" for _ in range(len(ax.get_xticks()))])
ax.set_yticks([0, 20, 40])
ax.set_title('QIF network variables')
ax.set_ylabel('\# neuron')
ax.set_xticklabels(["", "", ""])

# plot membrane potential trace
ax = fig.add_subplot(grid[1, :3])
ax.plot(data['v_i_rec'].squeeze(), c=to_hex(cmap[0]))
ax.set_ylabel(r'$v_i$')
ax.set_xlabel('time')
ax.set_ylim([-100, 100])
ax.set_xticklabels(["", "0", "20", "40"])

# plot qif average firing rate
ax = fig.add_subplot(grid[3, :3])
ax.plot(data['r_rec_av'].squeeze(), c=(0, 0, 0, 1))
ax.set_ylabel(r'$r$')
ax.set_title(r'$\frac{1}{N}\sum S_i$')
ax.set_yticks([0, 2, 4])
ax.set_xticklabels(["", "", ""])

# plot qif average membrane potential
ax = fig.add_subplot(grid[4, :3])
ax.plot(data['v_rec_av'].squeeze(), c=(0, 0, 0, 1))
ax.set_ylabel(r'$v$')
ax.set_title(r'$\frac{1}{N} \sum V_i$')
ax.set_yticks([-3, 0, 3])
ax.set_xticklabels(["", "0", "20", "40"])

# plot mf average firing rate
ax = fig.add_subplot(grid[3, 4:])
ax.plot(data['r_mac_rec_av'].squeeze(), c=to_hex(cmap[-1]))
ax.set_ylabel(r'$r$')
ax.set_title('Mean-field model variables')
ax.set_yticks([0, 2, 4])
ax.set_xticklabels(["", "", ""])

# plot qif average membrane potential
ax = fig.add_subplot(grid[4, 4:])
ax.plot(data['v_mac_rec_av'].squeeze(), c=to_hex(cmap[-1]))
ax.set_ylabel(r'$v$')
ax.set_yticks([-3, 0, 3])
ax.set_xticklabels(["", "0", "20", "40"])

# dummy axes
############

ax = fig.add_subplot(grid[0, 3:])
ax.set_xlabel('Microscopic state observation')
ax.set_title('Averaged microscopic observations')

ax = fig.add_subplot(grid[1, 3:])
ax.set_xlabel('Macroscopic state observation')
ax.set_title('QIF neuron network')

# final touches
###############

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving
fig.canvas.draw()
plt.savefig(f'neco_micro_macro.svg')
plt.show()
