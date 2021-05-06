import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.colors import to_hex
import scipy.io as scio
from pyrates.utility.pyauto import PyAuto
import sys
sys.path.append('../')

# preparations
##############

# plot settings
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (6.9, 3.0)
plt.rcParams['font.size'] = 8.0
plt.rcParams['axes.titlesize'] = 8.0
plt.rcParams['axes.labelsize'] = 8.0
plt.rcParams['lines.linewidth'] = 0.6
plt.rcParams['axes.titlepad'] = 1.0
labelpad = 1.0
plt.rcParams['axes.labelpad'] = labelpad
markersize = 15
cmap = sns.color_palette("plasma", as_cmap=False, n_colors=4)

# load auto data
path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"
fname = '../QIF_population/results/alpha_add.pkl'
a = PyAuto.from_file(fname, auto_dir=auto_dir)
a.update_bifurcation_style(bf_type='HB', color='k')

# load matlab data
matlab_path = "/home/rgast/MatlabProjects/QIFSimulations/matlab_scripts/neco"
data = scio.loadmat(f"{matlab_path}/neco_fig5_data.mat")

# plotting
##########

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)

# plot eta continuation for single alpha with limit cycle continuation
etas = [1.3]
ax = fig.add_subplot(grid[:, 0])
ax = a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta_3', ax=ax, line_color_stable=to_hex(cmap[0]),
                         line_color_unstable=to_hex(cmap[0]), default_size=markersize)
ax = a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta_hb2', ax=ax, ignore=['BP'], line_color_stable=to_hex(cmap[-1]),
                         default_size=markersize, custom_bf_styles={'LP': {'marker': 'p'}})
for eta in etas:
    plt.axvline(x=eta, color=(0, 0, 0, 1), linestyle='--')
ax.set_xlim([-6.0, 3.0])
ax.set_ylim([0., 4.0])
ax.set_xlabel(r'$\bar \eta$')
ax.set_ylabel(r'$r$')
ax.set_title('Steady-state and periodic solutions')

# plot the numerical integration results
inputs = [(800, 800), (2400, 800)]
title = r'$I(t) = 0.3$, $I(t) = -1.9$'
N = data['raster'].shape[1]
cutoff = int(N * 0.05)
n_neurons = 50
idx = np.random.randint(cutoff, N - cutoff, n_neurons)

# plot the firing rates
ax = fig.add_subplot(grid[0, 1])
ax.plot(data['r_rec_av'].squeeze(), c=(0, 0, 0, 1))
ax.plot(data['r_mac_rec_av'].squeeze(), c=to_hex(cmap[-1]))
ax_lims = ax.get_ylim()
for inp in inputs:
    r = Rectangle((inp[0], ax_lims[0]), width=inp[1], height=ax_lims[1] - ax_lims[0],
                  alpha=0.2, color='#7f7f7f')
    ax.add_patch(r)
ax.set_ylabel(r'$r$')
ax.set_title(title)
ax.set_yticks([0, 1, 2, 3])
ax.set_xticklabels(["", "", ""])
plt.legend(['QIF network', 'mean-field model'])

# plot the spikes
ax = fig.add_subplot(grid[1, 1])
ax.eventplot([data['raster'][0, j][:, 0] if len(data['raster'][0, j]) > 0 else np.asarray([])
              for j in idx], colors='k', lineoffsets=1.0, linelengths=1.0, linewidths=0.2)
ax.set_yticks([0, 20, 40])
ax.set_ylabel('\# neuron')
ax.set_xticklabels(["", "0", "100", "200", "300", "400"])
ax.set_xlabel('time')

# final touches
###############

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving
fig.canvas.draw()
plt.savefig(f'neco_sfa_1d_bf.svg')
plt.show()
