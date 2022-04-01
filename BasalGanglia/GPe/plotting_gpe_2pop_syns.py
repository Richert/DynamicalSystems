from matplotlib import gridspec
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_hex
import pickle
from pyrates.utility.pyauto import PyAuto
import sys
sys.path.append('../../')

# preparations
##############

fname = 'gpe_2pop_syns'

# load simulation data
data = pickle.load(open(f"results/{fname}.p", "rb"))

# plot settings
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (4.5, 2.5)
plt.rcParams['font.size'] = 8.0
plt.rcParams['axes.titlesize'] = 8.0
plt.rcParams['axes.labelsize'] = 8.0
plt.rcParams['lines.linewidth'] = 0.7
plt.rcParams['axes.titlepad'] = 2.0
labelpad = 2.0
plt.rcParams['axes.labelpad'] = labelpad
markersize = 15
cmap = sns.diverging_palette(145, 300, s=60, as_cmap=False, n=2)

# time window
idx_l, idx_u = (0, 5000)
t0, t1 = (0.0, 499.99)

############
# plotting #
############

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)

# time series plotting
######################

# timeseries for delay = 1.0
ax1 = fig.add_subplot(grid[0, 0])
ax1.plot(data['d0'].index[idx_l:idx_u], data['d0'].loc[t0:t1, 'r_i'], color=cmap[0])
# ax1.set_xlabel('time in ms')
ax1.set_ylabel('r')
ax1.set_title(r'$\mu = 1.0$, $\sigma^2 = 0.6$')
ax1.set_ylim([0.0, 0.125])
ax1.set_yticklabels(["0", "50", "100"])

ax2 = fig.add_subplot(grid[0, 1])
ax2.plot(data['d1'].index[idx_l:idx_u], data['d1'].loc[t0:t1, 'r_i'], color=cmap[0])
# ax2.set_xlabel('time in ms')
# ax2.set_ylabel('r')
ax2.set_title(r'$\mu = 2.0$, $\sigma^2 = 0.8$')
ax2.set_ylim([0.0, 0.125])
ax2.set_yticklabels(["0", "50", "100"])

ax3 = fig.add_subplot(grid[1, 0])
ax3.plot(data['d2'].index[idx_l:idx_u], data['d2'].loc[t0:t1, 'r_i'], color=cmap[0])
ax3.set_xlabel('time in ms')
ax3.set_ylabel('r')
ax3.set_title(r'$\mu = 3.0$, $\sigma^2 = 1.0$')
ax3.set_ylim([0.0, 0.125])
ax3.set_yticklabels(["0", "50", "100"])

ax4 = fig.add_subplot(grid[1, 1])
ax4.plot(data['d3'].index[idx_l:idx_u], data['d3'].loc[t0:t1, 'r_i'], color=cmap[0])
ax4.set_xlabel('time in ms')
# ax4.set_ylabel('r')
ax4.set_title(r'$\mu = 4.0$, $\sigma^2 = 1.2$')
ax4.set_ylim([0.0, 0.125])
ax4.set_yticklabels(["0", "50", "100"])

# final touches
###############

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving
fig.canvas.draw()
plt.savefig(f'results/{fname}_bf.svg')

plt.show()
